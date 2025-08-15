# DiffSensei-main/layout-generator/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layout_planner.heads import PredictionLoss

class StyleCalculator(nn.Module):
    """
    可微风格计算器（4维）：
      LD: sum(panel_area) / area_enclosing_rect  (min-max rectangle approx of panel vertices)
      AS: 1/(1 + var(cx)+var(cy)) for panel centers
      SI: mean RMS(offsets)
      BI: mean breakout_ratio weighted by element area (dialogs/chars)
    All operations are torch ops -> gradients flow.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("bonus", torch.tensor([0.0, 0.2, 0.4, 0.6, 0.6], dtype=torch.float32))

    @staticmethod
    def _area_xyxy01(xyxy):
        # xyxy: (...,4)
        if xyxy.numel() == 0:
            return xyxy.new_zeros(xyxy.shape[:-1])
        w = (xyxy[..., 2] - xyxy[..., 0]).clamp(min=0.0)
        h = (xyxy[..., 3] - xyxy[..., 1]).clamp(min=0.0)
        return w * h

    @staticmethod
    def _cxcy_from_xyxy01(xyxy):
        cx = 0.5 * (xyxy[..., 0] + xyxy[..., 2])
        cy = 0.5 * (xyxy[..., 1] + xyxy[..., 3])
        return cx, cy

    @staticmethod
    def _cxywh_to_xyxy(cxywh):
        # cxywh: (N,4) (cx,cy,w,h)
        if cxywh.numel() == 0:
            return cxywh.new_zeros((0,4))
        cx, cy, w, h = cxywh.unbind(-1)
        x1 = (cx - w / 2)
        y1 = (cy - h / 2)
        x2 = (cx + w / 2)
        y2 = (cy + h / 2)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, model_outputs, batch):
        """
        model_outputs: list of (p_out, d_out, c_out) for each sample
        batch: collated batch (dictionary) - used to get device and maybe other info
        returns: Tensor (B,4)
        """
        device = batch["style_vector"].device if "style_vector" in batch else torch.device("cpu")
        styles = []
        for idx, outs in enumerate(model_outputs):
            # unpack safely
            if outs is None:
                p_out = d_out = c_out = None
            else:
                # allow tuple or list of length 3
                try:
                    p_out, d_out, c_out = outs
                except Exception:
                    # unexpected shape -> treat all None
                    p_out = d_out = c_out = None

            # ensure we have tensors (or empty tensors)
            def get_tensor(outdict, key, default_shape=(0,)):
                if (outdict is None) or (not isinstance(outdict, dict)) or (key not in outdict):
                    return torch.zeros(default_shape, device=device)
                t = outdict[key]
                if t is None:
                    return torch.zeros(default_shape, device=device)
                return t.to(device)

            # Panels: expect xyxy (P,4) and offsets (P,8)
            p_bbox = get_tensor(p_out, "bbox", default_shape=(0,4))
            p_offsets = get_tensor(p_out, "offsets", default_shape=(p_bbox.shape[0] if p_bbox.numel() else 0, 8))

            # Dialogs & chars: expect bbox as cxywh (N,4), breakout_ratio (N,) or (N,1)
            d_bbox_cxywh = get_tensor(d_out, "bbox", default_shape=(0,4))
            d_break_ratio = get_tensor(d_out, "breakout_ratio", default_shape=(d_bbox_cxywh.shape[0] if d_bbox_cxywh.numel() else 0,))
            if d_break_ratio.ndim > 1:
                d_break_ratio = d_break_ratio.view(-1)

            c_bbox_cxywh = get_tensor(c_out, "bbox", default_shape=(0,4))
            c_break_ratio = get_tensor(c_out, "breakout_ratio", default_shape=(c_bbox_cxywh.shape[0] if c_bbox_cxywh.numel() else 0,))
            if c_break_ratio.ndim > 1:
                c_break_ratio = c_break_ratio.view(-1)

            # convert dialogs/chars bbox to xyxy for area calc
            d_xyxy = self._cxywh_to_xyxy(d_bbox_cxywh) if d_bbox_cxywh.numel() else d_bbox_cxywh
            c_xyxy = self._cxywh_to_xyxy(c_bbox_cxywh) if c_bbox_cxywh.numel() else c_bbox_cxywh

            # Device fallback
            if p_bbox.device != device:
                p_bbox = p_bbox.to(device)
            if p_offsets.device != device:
                p_offsets = p_offsets.to(device)
            if d_xyxy.device != device:
                d_xyxy = d_xyxy.to(device)
            if c_xyxy.device != device:
                c_xyxy = c_xyxy.to(device)
            if d_break_ratio.device != device:
                d_break_ratio = d_break_ratio.to(device)
            if c_break_ratio.device != device:
                c_break_ratio = c_break_ratio.to(device)

            # 1) layout_density: sum(panel_area) / enclosing_rect_area (min-max rect over panel bboxes)
            if p_bbox.numel():
                p_area = self._area_xyxy01(p_bbox)  # (P,)
                # enclosing rect (min x1, min y1, max x2, max y2)
                min_x = torch.min(p_bbox[..., 0])
                min_y = torch.min(p_bbox[..., 1])
                max_x = torch.max(p_bbox[..., 2])
                max_y = torch.max(p_bbox[..., 3])
                hull_area = ((max_x - min_x).clamp(min=0.0) * (max_y - min_y).clamp(min=0.0)).clamp(min=self.eps)
                layout_density = p_area.sum() / hull_area
            else:
                layout_density = torch.tensor(0.0, device=device)

            # 2) alignment_score: 1 / (1 + var(cx) + var(cy))
            if p_bbox.numel() and p_bbox.shape[0] > 1:
                cx, cy = self._cxcy_from_xyxy01(p_bbox)
                m = float(cx.shape[0])
                mean_cx = cx.mean()
                mean_cy = cy.mean()
                var_x = ((cx - mean_cx)**2).mean()
                var_y = ((cy - mean_cy)**2).mean()
                alignment_score = 1.0 / (1.0 + var_x + var_y + self.eps)
            else:
                alignment_score = torch.tensor(0.5, device=device)

            # 3) shape_instability: RMS over offsets
            if p_offsets.numel():
                rms = torch.sqrt((p_offsets**2).mean(dim=-1) + self.eps)  # (P,)
                shape_instability = rms.mean()
            else:
                shape_instability = torch.tensor(0.0, device=device)

            # 4) breakout_intensity: area-weighted mean of breakout ratios (dialogs + chars)
            all_areas = []
            all_ratios = []
            if d_xyxy.numel():
                d_area = self._area_xyxy01(d_xyxy)
                if d_break_ratio.numel() == d_area.numel():
                    all_areas.append(d_area)
                    all_ratios.append(d_break_ratio)
                else:
                    # mismatch lengths -> skip
                    pass
            if c_xyxy.numel():
                c_area = self._area_xyxy01(c_xyxy)
                if c_break_ratio.numel() == c_area.numel():
                    all_areas.append(c_area)
                    all_ratios.append(c_break_ratio)
                else:
                    pass

            if len(all_areas) > 0:
                areas_cat = torch.cat(all_areas, dim=0)
                ratios_cat = torch.cat(all_ratios, dim=0)
                denom = areas_cat.sum()
                if denom > 0:
                    breakout_intensity = (areas_cat * ratios_cat).sum() / (denom + self.eps)
                else:
                    # fallback to simple mean of ratios
                    breakout_intensity = ratios_cat.mean()
            else:
                breakout_intensity = torch.tensor(0.0, device=device)

            styles.append(torch.stack([layout_density, alignment_score, shape_instability, breakout_intensity], dim=0))

        if len(styles) == 0:
            return torch.zeros((0, 4), device=device)
        return torch.stack(styles, dim=0)  # (B,4)
    

class LayoutCompositeLoss(nn.Module):
    def __init__(self, lambda_style=0.1):
        super().__init__()
        self.lambda_style = float(lambda_style)
        self.pred_loss = PredictionLoss()
        self.style_calc = StyleCalculator()
        self.mse = nn.MSELoss()

    def forward(self, model_outputs, batch):
        """
        model_outputs: list of (panel_out, dialog_out, char_out) per sample
        batch: collated batch containing 'targets' and 'style_vector'
        """
        B = len(model_outputs)
        device = batch["style_vector"].device

        # geometric loss: sum per sample
        total_geom = torch.tensor(0.0, device=device)
        agg_loss_dict = {}
        for b_idx in range(B):
            p_out, d_out, c_out = model_outputs[b_idx]
            # per-sample targets are tensors from batch["targets"] (they are batched)
            targets_b = {k: v[b_idx] for k, v in batch["targets"].items()}
            loss_b, loss_dict_b = self.pred_loss((p_out, d_out, c_out), targets_b)
            total_geom = total_geom + loss_b
            # aggregate some entries for logging (sum)
            for k, v in loss_dict_b.items():
                agg_loss_dict.setdefault(k, 0.0)
                agg_loss_dict[k] += v

        geom_loss = total_geom / max(1, B)
        # average logging values
        for k in list(agg_loss_dict.keys()):
            agg_loss_dict[k] = agg_loss_dict[k] / max(1, B)

        # style loss: predicted style (B,4) vs batch["style_vector"] (B,4)
        style_pred = self.style_calc(model_outputs, batch)  # robust returns (B,4)
        # ensure shape aligns: if style_calc returned fewer entries or none, fallback
        if style_pred.numel() == 0:
            style_loss = torch.tensor(0.0, device=device)
        else:
            style_gt = batch["style_vector"].to(device)
            # if style_pred has different first-dim (shouldn't), align min len
            if style_pred.shape[0] != style_gt.shape[0]:
                L = min(style_pred.shape[0], style_gt.shape[0])
                style_loss = self.mse(style_pred[:L], style_gt[:L])
            else:
                style_loss = self.mse(style_pred, style_gt)

        total_loss = geom_loss + self.lambda_style * style_loss

        # build logs
        logs = {"geom_loss": float(geom_loss.item()), "style_loss": float(style_loss.item()), "total_loss": float(total_loss.item())}
        # merge per-component logs
        logs.update(agg_loss_dict)
        return total_loss, logs