import torch
import torch.nn as nn
import torch.nn.functional as F

# 工具函数：中心点+宽高 转 左上右下
def _cxywh_to_xyxy(cxywh):
    """
    Converts bounding boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2).
    Handles batched tensors of shape (..., 4).
    """
    if cxywh.numel() == 0:
        return cxywh.new_zeros(cxywh.shape)
    cx, cy, w, h = cxywh.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

class PredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use reduction='mean' for all losses, as masking will be applied before reduction
        self.ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1) # ignore_index for padded elements
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean') # Use mean reduction directly on valid elements

    def forward(self, predictions, batch):
        """
        Calculates prediction losses (classification and regression) for panels, dialogs, and characters.
        Assumes predictions and batch contain batched tensors with padding masks.

        predictions: A dictionary of batched tensors (B, N, D) or (B, N).
                     e.g., {'panel_bbox': (B, Pmax, 4), 'panel_offsets': (B, Pmax, 8),
                            'panel_class_logits': (B, Pmax, C), ...}
        batch: A dictionary of batched tensors (B, N, D) or (B, N).
               Expected to contain masks like 'panel_mask', 'dialog_mask', 'character_mask'
               and target values like 'panel_bboxes', 'panel_classes', etc.
        """
        # Determine device from an existing tensor in batch or predictions
        device = None
        if 'panel_mask' in batch and batch['panel_mask'] is not None:
            device = batch['panel_mask'].device
        elif 'panel_bbox' in predictions and predictions['panel_bbox'] is not None:
            device = predictions['panel_bbox'].device
        else:
            device = torch.device("cpu") # Fallback

        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # Helper to get masked tensors
        def get_masked_tensors(pred_key, target_key, mask_key):
            pred_tensor = predictions.get(pred_key)
            target_tensor = batch.get(target_key)
            mask_tensor = batch.get(mask_key)

            if pred_tensor is None or target_tensor is None or mask_tensor is None:
                return None, None, None

            # Apply mask
            masked_pred = pred_tensor[mask_tensor]
            masked_target = target_tensor[mask_tensor]
            
            # 修正形状，消除 (N,1) ↔ (N,) 警告
            if masked_pred.ndim == 2 and masked_pred.shape[1] == 1:
                masked_pred = masked_pred.squeeze(-1)
            if masked_target.ndim == 2 and masked_target.shape[1] == 1:
                masked_target = masked_target.squeeze(-1)
                
            return masked_pred, masked_target, mask_tensor.any()

        # --- Panels ---
        # Panel Class Loss (CE)
        masked_logits, masked_targets, is_valid = get_masked_tensors(
            'panel_class_logits', 'panel_classes', 'panel_mask'
        )
        if is_valid and masked_logits.numel() > 0:
            panel_class_loss = self.ce(masked_logits.view(-1, masked_logits.shape[-1]), masked_targets.view(-1))
            total_loss += panel_class_loss
            loss_dict['panel_class'] = panel_class_loss.item()

        # Panel Bbox L1 Loss (regression)
        masked_pred, masked_target, is_valid = get_masked_tensors(
            'panel_bbox', 'panel_bboxes', 'panel_mask'
        )
        if is_valid and masked_pred.numel() > 0:
            bbox_loss = self.l1(masked_pred, masked_target)
            total_loss += bbox_loss
            loss_dict['panel_bbox'] = bbox_loss.item()

        # Panel Offsets L1 Loss (regression)
        masked_pred, masked_target, is_valid = get_masked_tensors(
            'panel_offsets', 'panel_offsets', 'panel_mask'
        )
        if is_valid and masked_pred.numel() > 0:
            offsets_loss = self.l1(masked_pred, masked_target)
            total_loss += offsets_loss
            loss_dict['panel_offsets'] = offsets_loss.item()

        # --- Dialogs ---
        # Dialog Bbox L1 Loss
        masked_pred, masked_target, is_valid = get_masked_tensors(
            'dialog_bbox', 'dialog_bboxes', 'dialog_mask'
        )
        if is_valid and masked_pred.numel() > 0:
            dbbox_loss = self.l1(masked_pred, masked_target)
            total_loss += dbbox_loss
            loss_dict['dialog_bbox'] = dbbox_loss.item()

        # Dialog Breakout Class BCE Loss
        masked_logits, masked_targets, is_valid = get_masked_tensors(
            'dialog_breakout_logits', 'dialog_breakout_labels', 'dialog_mask'
        )
        if is_valid and masked_logits.numel() > 0:
            dbreak_loss = self.bce(masked_logits.view(-1), masked_targets.float().view(-1))
            total_loss += dbreak_loss
            loss_dict['dialog_breakout_class'] = dbreak_loss.item()

        # Dialog Breakout Ratio L1 Loss
        masked_pred, masked_target, is_valid = get_masked_tensors(
            'dialog_breakout_ratio', 'dialog_breakout_ratios', 'dialog_mask'
        )
        if is_valid and masked_pred.numel() > 0:
            dratio_loss = self.l1(masked_pred, masked_target)
            total_loss += dratio_loss
            loss_dict['dialog_breakout_ratio'] = dratio_loss.item()

        # Dialog Shape CE Loss
        masked_logits, masked_targets, is_valid = get_masked_tensors(
            'dialog_shape_logits', 'dialog_shapes', 'dialog_mask'
        )
        if is_valid and masked_logits.numel() > 0:
            dshape_loss = self.ce(masked_logits.view(-1, masked_logits.shape[-1]), masked_targets.view(-1))
            total_loss += dshape_loss
            loss_dict['dialog_shape'] = dshape_loss.item()

        # --- Characters ---
        # Character Bbox L1 Loss
        masked_pred, masked_target, is_valid = get_masked_tensors(
            'char_bbox', 'character_bboxes', 'character_mask'
        )
        if is_valid and masked_pred.numel() > 0:
            cbbox_loss = self.l1(masked_pred, masked_target)
            total_loss += cbbox_loss
            loss_dict['character_bbox'] = cbbox_loss.item()

        # Character Breakout Class BCE Loss
        masked_logits, masked_targets, is_valid = get_masked_tensors(
            'char_breakout_logits', 'character_breakout_labels', 'character_mask'
        )
        if is_valid and masked_logits.numel() > 0:
            cbreak_loss = self.bce(masked_logits.view(-1), masked_targets.float().view(-1))
            total_loss += cbreak_loss
            loss_dict['character_breakout_class'] = cbreak_loss.item()

        # Character Breakout Ratio L1 Loss
        masked_pred, masked_target, is_valid = get_masked_tensors(
            'char_breakout_ratio', 'character_breakout_ratios', 'character_mask'
        )
        if is_valid and masked_pred.numel() > 0:
            cratio_loss = self.l1(masked_pred, masked_target)
            total_loss += cratio_loss
            loss_dict['character_breakout_ratio'] = cratio_loss.item()

        return total_loss, loss_dict
    
class StyleCalculator(nn.Module):
    """
    Differentiable style calculator (4 dimensions) - Vectorized version
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _area_xyxy01(xyxy):
        x1, y1, x2, y2 = xyxy.unbind(-1)
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        return w * h

    @staticmethod
    def _cxcy_from_xyxy01(xyxy):
        cx = 0.5 * (xyxy[..., 0] + xyxy[..., 2])
        cy = 0.5 * (xyxy[..., 1] + xyxy[..., 3])
        return cx, cy

    def forward(self, predictions, batch):
        device = batch['panel_mask'].device if 'panel_mask' in batch else torch.device("cpu")
        B = batch['panel_mask'].shape[0]

        # Panels
        panel_mask = batch['panel_mask']  # (B, Pmax)
        p_bbox_xyxy = _cxywh_to_xyxy(predictions['panel_bbox'])  # (B, Pmax, 4)
        p_offsets = predictions['panel_offsets']  # (B, Pmax, 8)

        # 1) layout_density
        p_area = self._area_xyxy01(p_bbox_xyxy) * panel_mask  # (B, Pmax)
        min_x = torch.where(panel_mask, p_bbox_xyxy[..., 0], 1.0)
        min_y = torch.where(panel_mask, p_bbox_xyxy[..., 1], 1.0)
        max_x = torch.where(panel_mask, p_bbox_xyxy[..., 2], 0.0)
        max_y = torch.where(panel_mask, p_bbox_xyxy[..., 3], 0.0)

        min_x = min_x.masked_fill(~panel_mask, float('inf')).amin(dim=1)
        min_y = min_y.masked_fill(~panel_mask, float('inf')).amin(dim=1)
        max_x = max_x.masked_fill(~panel_mask, float('-inf')).amax(dim=1)
        max_y = max_y.masked_fill(~panel_mask, float('-inf')).amax(dim=1)

        hull_w = (max_x - min_x).clamp(min=0.0)
        hull_h = (max_y - min_y).clamp(min=0.0)
        hull_area = (hull_w * hull_h).clamp(min=self.eps)

        layout_density = p_area.sum(dim=1) / hull_area  # (B,)

        # 2) alignment_score (mask 少于2个时给0.5)
        valid_counts = panel_mask.sum(dim=1)
        cx, cy = self._cxcy_from_xyxy01(p_bbox_xyxy)
        cx = torch.where(panel_mask, cx, torch.nan)
        cy = torch.where(panel_mask, cy, torch.nan)
        var_x = torch.nanvar(cx, dim=1)
        var_y = torch.nanvar(cy, dim=1)
        alignment_score = torch.where(valid_counts > 1, 1.0 / (1.0 + var_x + var_y + self.eps),
                                      torch.full_like(var_x, 0.5))

        # 3) shape_instability
        rms_per_panel = torch.sqrt((p_offsets ** 2).mean(dim=-1) + self.eps)
        rms_per_panel = torch.where(panel_mask, rms_per_panel, torch.nan)
        shape_instability = torch.nanmean(rms_per_panel, dim=1).nan_to_num(0.0)

        # 4) breakout_intensity (批处理)
        def process_breakout(bbox_tensor, ratio_tensor, mask_tensor):
            xyxy = _cxywh_to_xyxy(bbox_tensor)
            area = self._area_xyxy01(xyxy)  # (B, N)
            ratio = ratio_tensor.squeeze(-1) if ratio_tensor.ndim > 2 else ratio_tensor
            area = torch.where(mask_tensor, area, 0.0)
            ratio = torch.where(mask_tensor, ratio, 0.0)
            return area, ratio

        d_area, d_ratio = process_breakout(predictions['dialog_bbox'],
                                           predictions['dialog_breakout_ratio'],
                                           batch['dialog_mask'])
        c_area, c_ratio = process_breakout(predictions['char_bbox'],
                                           predictions['char_breakout_ratio'],
                                           batch['character_mask'])
        all_area = torch.cat([d_area, c_area], dim=1)  # (B, Dmax+Cmax)
        all_ratio = torch.cat([d_ratio, c_ratio], dim=1)
        denom = all_area.sum(dim=1)
        breakout_intensity = torch.where(
            denom > self.eps,
            (all_area * all_ratio).sum(dim=1) / denom,
            torch.where(all_ratio.sum(dim=1) > 0, all_ratio.mean(dim=1), torch.zeros_like(denom))
        )

        # Stack batch style vector
        style_pred = torch.stack([
            layout_density,
            alignment_score,
            shape_instability,
            breakout_intensity
        ], dim=-1)
        return style_pred

class GeometricConstraintLoss(nn.Module):
    """
    Vectorized geometric constraint loss (Overlap Loss + Containment Loss)
    using batch computations instead of per-sample loops.
    """
    def __init__(self, breakout_thresh=0.02, eps=1e-6):
        super().__init__()
        self.breakout_thresh = breakout_thresh
        self.eps = eps

    def _batch_iou_self(self, boxes, mask):
        """
        Efficient batch IoU for sets of boxes within each sample.
        boxes: (B, N, 4), mask: (B, N) bool
        Output: sum of upper triangle IoUs per batch, shape (B,)
        """
        B, N, _ = boxes.shape
        mask2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, N, N)
        lt = torch.max(boxes[:, :, None, :2], boxes[:, None, :, :2])  # (B, N, N, 2)
        rb = torch.min(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
        union = area[:, :, None] + area[:, None, :] - inter
        iou = inter / (union + self.eps)
        # Mask out invalid pairs (including self)
        iou = torch.where(mask2d, iou, torch.zeros_like(iou))
        triu_mask = torch.triu(torch.ones((N, N), device=boxes.device), diagonal=1).bool()
        triu_mask = triu_mask.unsqueeze(0)  # broadcast to batch
        iou = torch.where(triu_mask, iou, torch.zeros_like(iou))
        return iou.sum(dim=(1, 2))  # (B,)

    def _containment_loss_batch(self, child_xyxy, parent_xyxy, breakout_ratio, valid_mask):
        """
        Vector batch containment loss
        All inputs (B, M, 4) except breakout_ratio (B, M) and valid_mask (B, M)
        """
        lt = torch.max(child_xyxy[..., :2], parent_xyxy[..., :2])
        rb = torch.min(child_xyxy[..., 2:], parent_xyxy[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        child_area = (child_xyxy[..., 2] - child_xyxy[..., 0]) * (child_xyxy[..., 3] - child_xyxy[..., 1])
        outside_area = (child_area - inter).clamp(min=0)
        loss_per_element = outside_area / (child_area + self.eps)
        mask = (breakout_ratio < self.breakout_thresh) & valid_mask
        if mask.any():
            return loss_per_element[mask].mean()
        else:
            return torch.tensor(0.0, device=child_xyxy.device)

    def forward(self, predictions, batch):
        device = batch['panel_mask'].device
        B = batch['panel_mask'].shape[0]

        # Preprocess panel data
        panel_mask = batch['panel_mask']  # (B, Pmax)
        panel_bboxes_xyxy = _cxywh_to_xyxy(predictions['panel_bbox'])

        # Overlap loss: batch computation
        overlap_loss_batch = torch.zeros(B, device=device)
        multi_panel_mask = panel_mask.sum(dim=1) > 1
        if multi_panel_mask.any():
            overlap_loss_batch[multi_panel_mask] = self._batch_iou_self(panel_bboxes_xyxy, panel_mask)[multi_panel_mask]

        # Containment: dialogs
        dialog_mask = batch['dialog_mask']  # (B, Dmax)
        dialog_parent_idx = batch['dialog_parent_idx']
        dialog_child_bbox = _cxywh_to_xyxy(predictions['dialog_bbox'])
        dialog_ratio = predictions['dialog_breakout_ratio'].squeeze(-1) if predictions['dialog_breakout_ratio'].dim() > 2 else predictions['dialog_breakout_ratio']

        # Map parent idx to parent boxes
        dialog_parent_boxes = torch.gather(panel_bboxes_xyxy, 1, dialog_parent_idx.unsqueeze(-1).expand(-1, -1, 4))
        dialog_loss = self._containment_loss_batch(
            dialog_child_bbox, dialog_parent_boxes, dialog_ratio, dialog_mask
        )

        # Containment: chars
        char_mask = batch['character_mask']
        char_parent_idx = batch['char_parent_idx']
        char_child_bbox = _cxywh_to_xyxy(predictions['char_bbox'])
        char_ratio = predictions['char_breakout_ratio'].squeeze(-1) if predictions['char_breakout_ratio'].dim() > 2 else predictions['char_breakout_ratio']
        char_parent_boxes = torch.gather(panel_bboxes_xyxy, 1, char_parent_idx.unsqueeze(-1).expand(-1, -1, 4))
        char_loss = self._containment_loss_batch(
            char_child_bbox, char_parent_boxes, char_ratio, char_mask
        )

        # Mean over batch
        overlap_loss = overlap_loss_batch.mean()
        containment_loss = dialog_loss + char_loss
        total_geom_loss = overlap_loss + containment_loss

        loss_dict = {
            "geom_overlap_loss": overlap_loss.item(),
            "geom_containment_loss": containment_loss.item()
        }
        return total_geom_loss, loss_dict

class LayoutCompositeLoss(nn.Module):
    def __init__(self, lambda_style=0.1, lambda_geom=0.1):
        super().__init__()
        self.lambda_style = lambda_style
        self.lambda_geom = lambda_geom
        self.pred_loss_fn = PredictionLoss()            # 已经批处理
        self.style_calc_fn = StyleCalculator()          # 已向量化
        self.geom_loss_fn = GeometricConstraintLoss()   # 新批处理版
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, predictions, batch):
        device = batch['panel_mask'].device

        # PredictionLoss
        pred_loss, pred_loss_dict = self.pred_loss_fn(predictions, batch)

        # StyleLoss
        style_pred_values = self.style_calc_fn(predictions, batch)  # (B, 4)
        style_gt = batch["style_vector"].to(device)                 # (B, 4)
        style_loss = self.mse(style_pred_values, style_gt)

        # Geometric Loss
        geom_loss, geom_loss_dict = self.geom_loss_fn(predictions, batch)

        total_loss = pred_loss + self.lambda_style * style_loss + self.lambda_geom * geom_loss
        loss_dict = {
            **pred_loss_dict,
            "style_loss": style_loss.item(),
            **geom_loss_dict,
            "geom_loss": geom_loss.item(),
            "pred_loss": pred_loss.item(),
            "total_loss": total_loss.item()
        }
        return total_loss, loss_dict
