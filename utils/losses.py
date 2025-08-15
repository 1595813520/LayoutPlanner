# DiffSensei-main/layout-generator/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layout_planner.heads import PredictionLoss
class StyleCalculator(nn.Module):
    """
    可微风格计算器（4维）：
      LD: sum(panel_area) / area_enclosing_rect  (uses panel xyxy)
      AS: 1/(1 + var(cx)+var(cy)) for panel centers
      SI: mean RMS(offsets)
      BI: mean breakout_ratio weighted by element area (dialogs/chars)
    注意：model_outputs 中的 panel bbox 我们期望为 cxywh（模型 heads 输出），
    因此需要先将其转为 xyxy 再做面积/中心等计算。
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # self.register_buffer("bonus", torch.tensor([0.0, 0.2, 0.4, 0.6, 0.6], dtype=torch.float32))
        # （未使用的 buffer 注释掉，避免困惑）

    @staticmethod
    def _area_xyxy01(xyxy):
        # xyxy: (...,4)
        if xyxy.numel() == 0:
            return xyxy.new_zeros(xyxy.shape[:-1])
        x1 = xyxy[..., 0]
        y1 = xyxy[..., 1]
        x2 = xyxy[..., 2]
        y2 = xyxy[..., 3]
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        return w * h

    @staticmethod
    def _cxcy_from_xyxy01(xyxy):
        cx = 0.5 * (xyxy[..., 0] + xyxy[..., 2])
        cy = 0.5 * (xyxy[..., 1] + xyxy[..., 3])
        return cx, cy

    @staticmethod
    def _cxywh_to_xyxy(cxywh):
        # cxywh: (...,4) (cx,cy,w,h) -> xyxy
        if cxywh.numel() == 0:
            return cxywh.new_zeros((0, 4))
        cx = cxywh[..., 0]
        cy = cxywh[..., 1]
        w = cxywh[..., 2]
        h = cxywh[..., 3]
        x1 = (cx - w / 2.0)
        y1 = (cy - h / 2.0)
        x2 = (cx + w / 2.0)
        y2 = (cy + h / 2.0)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _ensure_tensor(self, x, device, default_shape=(0,)):
        # 保证返回 tensor（并移动到 device）
        if isinstance(x, torch.Tensor):
            return x.to(device)
        try:
            return torch.as_tensor(x, device=device)
        except Exception:
            return torch.zeros(default_shape, device=device)

    def forward(self, model_outputs, batch):
        device = batch["style_vector"].device if "style_vector" in batch else torch.device("cpu")
        styles = []
        for idx, outs in enumerate(model_outputs):
            # 默认空 dict 的处理
            if outs is None:
                p_out = d_out = c_out = {}
            else:
                try:
                    p_out, d_out, c_out = outs
                except Exception:
                    p_out = d_out = c_out = {}

            def get_tensor(outdict, key, default_shape=(0,)):
                if (outdict is None) or (not isinstance(outdict, dict)) or (key not in outdict):
                    return torch.zeros(default_shape, device=device)
                t = outdict[key]
                if t is None:
                    return torch.zeros(default_shape, device=device)
                # 保证为 tensor
                return self._ensure_tensor(t, device, default_shape)

            # Panels: model 输出为 cxywh (num_panels,4)， offsets (num_panels,8)
            p_bbox_cxywh = get_tensor(p_out, "bbox", default_shape=(0,4))
            p_offsets = get_tensor(p_out, "offsets", default_shape=(p_bbox_cxywh.shape[0] if p_bbox_cxywh.numel() else 0, 8))

            # 若是 (N,1) 形式的 breakout_ratio，把它 squeeze 成 (N,)
            d_bbox_cxywh = get_tensor(d_out, "bbox", default_shape=(0,4))
            d_break_ratio = get_tensor(d_out, "breakout_ratio", default_shape=(d_bbox_cxywh.shape[0] if d_bbox_cxywh.numel() else 0,))
            if isinstance(d_break_ratio, torch.Tensor) and d_break_ratio.ndim > 1:
                d_break_ratio = d_break_ratio.view(-1)

            c_bbox_cxywh = get_tensor(c_out, "bbox", default_shape=(0,4))
            c_break_ratio = get_tensor(c_out, "breakout_ratio", default_shape=(c_bbox_cxywh.shape[0] if c_bbox_cxywh.numel() else 0,))
            if isinstance(c_break_ratio, torch.Tensor) and c_break_ratio.ndim > 1:
                c_break_ratio = c_break_ratio.view(-1)

            # convert dialogs/chars bbox to xyxy for area calc (they are cxywh)
            d_xyxy = self._cxywh_to_xyxy(d_bbox_cxywh) if d_bbox_cxywh.numel() else d_bbox_cxywh
            c_xyxy = self._cxywh_to_xyxy(c_bbox_cxywh) if c_bbox_cxywh.numel() else c_bbox_cxywh

            # 对 panels：模型输出是 cxywh（heads），因此转换为 xyxy
            if p_bbox_cxywh.numel():
                p_bbox_xyxy = self._cxywh_to_xyxy(p_bbox_cxywh)
            else:
                p_bbox_xyxy = p_bbox_cxywh  # 空 tensor

            # move to device（保证）
            p_bbox_xyxy = p_bbox_xyxy.to(device)
            p_offsets = p_offsets.to(device)
            d_xyxy = d_xyxy.to(device)
            c_xyxy = c_xyxy.to(device)
            if isinstance(d_break_ratio, torch.Tensor):
                d_break_ratio = d_break_ratio.to(device)
            else:
                d_break_ratio = torch.zeros((0,), device=device)
            if isinstance(c_break_ratio, torch.Tensor):
                c_break_ratio = c_break_ratio.to(device)
            else:
                c_break_ratio = torch.zeros((0,), device=device)

            # 1) layout_density: sum(panel_area) / enclosing_rect_area
            if p_bbox_xyxy.numel():
                p_area = self._area_xyxy01(p_bbox_xyxy)  # (P,)
                min_x = torch.min(p_bbox_xyxy[..., 0])
                min_y = torch.min(p_bbox_xyxy[..., 1])
                max_x = torch.max(p_bbox_xyxy[..., 2])
                max_y = torch.max(p_bbox_xyxy[..., 3])
                hull_w = (max_x - min_x).clamp(min=0.0)
                hull_h = (max_y - min_y).clamp(min=0.0)
                hull_area = (hull_w * hull_h).clamp(min=self.eps)
                layout_density = p_area.sum() / hull_area
            else:
                layout_density = torch.tensor(0.0, device=device)

            # 2) alignment_score: 1 / (1 + var(cx) + var(cy))
            if p_bbox_xyxy.numel() and p_bbox_xyxy.shape[0] > 1:
                cx, cy = self._cxcy_from_xyxy01(p_bbox_xyxy)
                var_x = ((cx - cx.mean())**2).mean()
                var_y = ((cy - cy.mean())**2).mean()
                alignment_score = 1.0 / (1.0 + var_x + var_y + self.eps)
            else:
                alignment_score = torch.tensor(0.5, device=device)

            # 3) shape_instability: RMS over offsets
            if p_offsets.numel():
                # offsets shape expected (P,8)
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
            if c_xyxy.numel():
                c_area = self._area_xyxy01(c_xyxy)
                if c_break_ratio.numel() == c_area.numel():
                    all_areas.append(c_area)
                    all_ratios.append(c_break_ratio)

            if len(all_areas) > 0:
                areas_cat = torch.cat(all_areas, dim=0)
                ratios_cat = torch.cat(all_ratios, dim=0)
                denom = areas_cat.sum()
                if denom > 0:
                    breakout_intensity = (areas_cat * ratios_cat).sum() / (denom + self.eps)
                else:
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
        device = batch["style_vector"].device if "style_vector" in batch else torch.device("cpu")

        # geometric loss: sum per sample
        total_geom = torch.tensor(0.0, device=device)
        agg_loss_dict = {}
        for b_idx in range(B):
            outs = model_outputs[b_idx]
            # 安全展开（有些实现可能返回 None）
            if outs is None:
                p_out = d_out = c_out = {}
            else:
                try:
                    p_out, d_out, c_out = outs
                except Exception:
                    p_out = d_out = c_out = {}

            targets_b = {k: v[b_idx] for k, v in batch["targets"].items()}
            loss_b, loss_dict_b = self.pred_loss((p_out, d_out, c_out), targets_b)
            total_geom = total_geom + loss_b
            for k, v in loss_dict_b.items():
                agg_loss_dict.setdefault(k, 0.0)
                agg_loss_dict[k] += v

        geom_loss = total_geom / max(1, B)
        for k in list(agg_loss_dict.keys()):
            agg_loss_dict[k] = agg_loss_dict[k] / max(1, B)

        # style loss
        style_pred = self.style_calc(model_outputs, batch)  # (B,4) or empty
        if style_pred.numel() == 0:
            style_loss = torch.tensor(0.0, device=device)
        else:
            style_gt = batch["style_vector"].to(device)
            if style_pred.shape[0] != style_gt.shape[0]:
                L = min(style_pred.shape[0], style_gt.shape[0])
                style_loss = self.mse(style_pred[:L], style_gt[:L])
            else:
                style_loss = self.mse(style_pred, style_gt)

        total_loss = geom_loss + self.lambda_style * style_loss

        logs = {"geom_loss": float(geom_loss.item()), "style_loss": float(style_loss.item()), "total_loss": float(total_loss.item())}
        logs.update(agg_loss_dict)
        return total_loss, logs