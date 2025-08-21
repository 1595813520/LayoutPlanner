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

# --- REVISED PredictionLoss (fully batched, includes all original losses) ---
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

# --- REVISED StyleCalculator (processes batched inputs, calculates per-sample styles) ---
class StyleCalculator(nn.Module):
    """
    Differentiable style calculator (4 dimensions):
      LD: sum(panel_area) / area_enclosing_rect  (uses panel xyxy)
      AS: 1/(1 + var(cx)+var(cy)) for panel centers
      SI: mean RMS(offsets)
      BI: mean breakout_ratio weighted by element area (dialogs/chars)

    Assumes model_outputs (predictions) contains batched tensors (B, N, D) or (B, N).
    Calculates style metrics for each sample in the batch.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _area_xyxy01(xyxy):
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

    def forward(self, predictions, batch):
        """
        predictions: A dictionary of batched tensors (B, N, D) or (B, N).
        batch: A dictionary of batched tensors containing masks like 'panel_mask', 'dialog_mask', 'character_mask'.
        Returns: (B, 4) tensor of style values for the batch.
        """
        # Determine device from an existing tensor in batch or predictions
        device = None
        if 'panel_mask' in batch and batch['panel_mask'] is not None:
            device = batch['panel_mask'].device
        elif 'panel_bbox' in predictions and predictions['panel_bbox'] is not None:
            device = predictions['panel_bbox'].device
        else:
            device = torch.device("cpu") # Fallback

        B = batch['panel_mask'].shape[0] # Assume panel_mask is always present and defines batch size

        # Initialize style components for the batch
        layout_densities = torch.zeros(B, device=device)
        alignment_scores = torch.zeros(B, device=device)
        shape_instabilities = torch.zeros(B, device=device)
        breakout_intensities = torch.zeros(B, device=device)

        # Get relevant batched tensors from predictions and batch
        panel_mask_batch = batch.get('panel_mask') # (B, Pmax)
        dialog_mask_batch = batch.get('dialog_mask') # (B, Dmax)
        character_mask_batch = batch.get('character_mask') # (B, Cmax)

        p_bbox_cxywh_batch = predictions.get('panel_bbox') # (B, Pmax, 4)
        p_offsets_batch = predictions.get('panel_offsets') # (B, Pmax, 8)
        d_bbox_cxywh_batch = predictions.get('dialog_bbox') # (B, Dmax, 4)
        d_break_ratio_batch = predictions.get('dialog_breakout_ratio') # (B, Dmax, 1) or (B, Dmax)
        c_bbox_cxywh_batch = predictions.get('char_bbox') # (B, Cmax, 4)
        c_break_ratio_batch = predictions.get('char_breakout_ratio') # (B, Cmax, 1) or (B, Cmax)

        # 确保 mask 与预测对齐
        if panel_mask_batch is not None and p_bbox_cxywh_batch is not None:
            pb_cols = p_bbox_cxywh_batch.shape[1]
            if panel_mask_batch.shape[1] != pb_cols:
                if panel_mask_batch.shape[1] > pb_cols:
                    # 截断多余列（通常是 token mask 的情况）
                    panel_mask_batch = panel_mask_batch[:, :pb_cols]
                else:
                    # 补 False
                    pad_cols = pb_cols - panel_mask_batch.shape[1]
                    pad = torch.zeros((panel_mask_batch.shape[0], pad_cols),
                                    dtype=panel_mask_batch.dtype,
                                    device=panel_mask_batch.device)
                    panel_mask_batch = torch.cat([panel_mask_batch, pad], dim=1)
                    
        # Ensure breakout ratios are (B, N)
        if d_break_ratio_batch is not None and d_break_ratio_batch.ndim > 2:
            d_break_ratio_batch = d_break_ratio_batch.squeeze(-1)
        if c_break_ratio_batch is not None and c_break_ratio_batch.ndim > 2:
            c_break_ratio_batch = c_break_ratio_batch.squeeze(-1)

        # Iterate over each sample in the batch to calculate style metrics
        for b_idx in range(B):
            # Panel-related calculations for current sample
            current_panel_mask = panel_mask_batch[b_idx]
            if current_panel_mask.any() and p_bbox_cxywh_batch is not None and p_offsets_batch is not None:
                p_bbox_xyxy = _cxywh_to_xyxy(p_bbox_cxywh_batch[b_idx][current_panel_mask])
                current_p_offsets = p_offsets_batch[b_idx][current_panel_mask]

                # 1) layout_density
                p_area = self._area_xyxy01(p_bbox_xyxy)
                min_x = torch.min(p_bbox_xyxy[..., 0])
                min_y = torch.min(p_bbox_xyxy[..., 1])
                max_x = torch.max(p_bbox_xyxy[..., 2])
                max_y = torch.max(p_bbox_xyxy[..., 3])
                hull_w = (max_x - min_x).clamp(min=0.0)
                hull_h = (max_y - min_y).clamp(min=0.0)
                hull_area = (hull_w * hull_h).clamp(min=self.eps)
                layout_densities[b_idx] = p_area.sum() / hull_area

                # 2) alignment_score
                if p_bbox_xyxy.shape[0] > 1: # Need at least 2 panels for variance
                    cx, cy = self._cxcy_from_xyxy01(p_bbox_xyxy)
                    var_x = ((cx - cx.mean())**2).mean()
                    var_y = ((cy - cy.mean())**2).mean()
                    alignment_scores[b_idx] = 1.0 / (1.0 + var_x + var_y + self.eps)
                else:
                    alignment_scores[b_idx] = torch.tensor(0.5, device=device) # Default for single/no panels

                # 3) shape_instability
                if current_p_offsets.numel():
                    rms = torch.sqrt((current_p_offsets**2).mean(dim=-1) + self.eps)
                    shape_instabilities[b_idx] = rms.mean()
                else:
                    shape_instabilities[b_idx] = torch.tensor(0.0, device=device)
            else: # No panels in this sample
                layout_densities[b_idx] = torch.tensor(0.0, device=device)
                alignment_scores[b_idx] = torch.tensor(0.5, device=device)
                shape_instabilities[b_idx] = torch.tensor(0.0, device=device)

            # Breakout intensity for current sample
            all_areas = []
            all_ratios = []

            current_dialog_mask = dialog_mask_batch[b_idx]
            if current_dialog_mask.any() and d_bbox_cxywh_batch is not None and d_break_ratio_batch is not None:
                d_xyxy = _cxywh_to_xyxy(d_bbox_cxywh_batch[b_idx][current_dialog_mask])
                current_d_break_ratio = d_break_ratio_batch[b_idx][current_dialog_mask]
                d_area = self._area_xyxy01(d_xyxy)
                if d_area.numel() == current_d_break_ratio.numel(): # Ensure sizes match
                    all_areas.append(d_area)
                    all_ratios.append(current_d_break_ratio)

            current_character_mask = character_mask_batch[b_idx]
            if current_character_mask.any() and c_bbox_cxywh_batch is not None and c_break_ratio_batch is not None:
                c_xyxy = _cxywh_to_xyxy(c_bbox_cxywh_batch[b_idx][current_character_mask])
                current_c_break_ratio = c_break_ratio_batch[b_idx][current_character_mask]
                c_area = self._area_xyxy01(c_xyxy)
                if c_area.numel() == current_c_break_ratio.numel(): # Ensure sizes match
                    all_areas.append(c_area)
                    all_ratios.append(current_c_break_ratio)

            if len(all_areas) > 0:
                areas_cat = torch.cat(all_areas, dim=0)
                ratios_cat = torch.cat(all_ratios, dim=0)
                denom = areas_cat.sum()
                if denom > self.eps: # Use eps for denominator to prevent division by zero
                    breakout_intensities[b_idx] = (areas_cat * ratios_cat).sum() / denom
                elif ratios_cat.numel() > 0: # If total area is zero but there are elements, take mean ratio
                    breakout_intensities[b_idx] = ratios_cat.mean()
                else:
                    breakout_intensities[b_idx] = torch.tensor(0.0, device=device)
            else:
                breakout_intensities[b_idx] = torch.tensor(0.0, device=device)

        # Stack all style values for the batch
        style_pred = torch.stack([layout_densities, alignment_scores, shape_instabilities, breakout_intensities], dim=1) # (B, 4)
        
        # StyleCalculator returns the predicted style values, MSELoss will be calculated in CombinedLoss
        return style_pred

# --- GeometricConstraintLoss (already mostly correct, minor adjustments for robustness) ---
class GeometricConstraintLoss(nn.Module):
    """
    Geometric constraint loss (Overlap Loss and Containment Loss).
    Operates on batched inputs, calculating per-sample losses and then averaging over the batch.
    """
    def __init__(self, breakout_thresh=0.02, eps=1e-6):
        super().__init__()
        self.breakout_thresh = breakout_thresh
        self.eps = eps

    def _calculate_iou_matrix(self, boxes1, boxes2):
        """
        Calculates Intersection over Union (IoU) matrix between two sets of bounding boxes.
        boxes1: (N, 4), boxes2: (M, 4) in (x1, y1, x2, y2) format.
        Returns: (N, M) IoU matrix.
        """
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - inter
        return inter / (union + self.eps)

    def _containment_loss(self, child_xyxy, parent_xyxy, breakout_ratio):
        """
        Calculates containment loss for child elements (dialogs/chars) within their parent panels.
        Loss is applied only if breakout_ratio is below breakout_thresh.
        child_xyxy: (N, 4) in (x1, y1, x2, y2) format.
        parent_xyxy: (N, 4) in (x1, y1, x2, y2) format (aligned parents for each child).
        breakout_ratio: (N,) tensor of breakout ratios for each child.
        """
        lt = torch.max(child_xyxy[:, :2], parent_xyxy[:, :2])
        rb = torch.min(child_xyxy[:, 2:], parent_xyxy[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        child_area = (child_xyxy[:, 2] - child_xyxy[:, 0]) * (child_xyxy[:, 3] - child_xyxy[:, 1])
        outside_area = (child_area - inter).clamp(min=0) # Area of child outside parent
        
        # Loss is proportion of child area outside parent
        loss_per_element = outside_area / (child_area + self.eps)
        
        # Apply mask: only compute loss for elements with breakout_ratio < threshold
        mask = (breakout_ratio < self.breakout_thresh)
        
        # Ensure mask is boolean and has same device as loss_per_element
        if not mask.dtype == torch.bool:
            mask = mask.bool()
        
        if mask.any():
            return loss_per_element[mask].mean()
        else:
            return torch.tensor(0.0, device=child_xyxy.device)

    def forward(self, predictions, batch):
        """
        Calculates panel overlap loss and child element containment loss for a batch.
        predictions: A dictionary of batched tensors (B, N, D) or (B, N).
        batch: A dictionary of batched tensors containing masks and parent indices.
        """
        # Determine device
        device = None
        if 'panel_mask' in batch and batch['panel_mask'] is not None:
            device = batch['panel_mask'].device
        elif 'panel_bbox' in predictions and predictions['panel_bbox'] is not None:
            device = predictions['panel_bbox'].device
        else:
            device = torch.device("cpu") # Fallback

        B = batch['panel_mask'].shape[0] # Assume panel_mask defines batch size

        total_overlap_loss_sum = torch.tensor(0.0, device=device)
        total_containment_loss_sum = torch.tensor(0.0, device=device)

        # Get batched tensors (predictions and masks/indices from batch)
        panel_bboxes_cxywh_batch = predictions.get('panel_bbox') # (B, Pmax, 4)
        panel_mask_batch = batch.get('panel_mask') # (B, Pmax)

        dialog_bboxes_cxywh_batch = predictions.get('dialog_bbox') # (B, Dmax, 4)
        dialog_breakout_ratio_batch = predictions.get('dialog_breakout_ratio') # (B, Dmax, 1) or (B, Dmax)
        dialog_parent_idx_batch = batch.get('dialog_parent_idx') # (B, Dmax)
        dialog_mask_batch = batch.get('dialog_mask') # (B, Dmax)

        char_bboxes_cxywh_batch = predictions.get('char_bbox') # (B, Cmax, 4)
        char_breakout_ratio_batch = predictions.get('char_breakout_ratio') # (B, Cmax, 1) or (B, Cmax)
        char_parent_idx_batch = batch.get('char_parent_idx') # (B, Cmax)
        character_mask_batch = batch.get('character_mask') # (B, Cmax)

        # Ensure breakout ratios are (B, N)
        if dialog_breakout_ratio_batch is not None and dialog_breakout_ratio_batch.ndim > 2:
            dialog_breakout_ratio_batch = dialog_breakout_ratio_batch.squeeze(-1)
        if char_breakout_ratio_batch is not None and char_breakout_ratio_batch.ndim > 2:
            char_breakout_ratio_batch = char_breakout_ratio_batch.squeeze(-1)

        for b in range(B):
            # --- Panel Overlap Loss (per sample) ---
            current_panel_mask = panel_mask_batch[b]
            if current_panel_mask.any() and panel_bboxes_cxywh_batch is not None:
                panel_bboxes_cxywh_sample = panel_bboxes_cxywh_batch[b][current_panel_mask]
                if panel_bboxes_cxywh_sample.shape[0] > 1: # Need at least 2 panels for overlap
                    panel_bboxes_xyxy_sample = _cxywh_to_xyxy(panel_bboxes_cxywh_sample)
                    iou_mat = self._calculate_iou_matrix(panel_bboxes_xyxy_sample, panel_bboxes_xyxy_sample)
                    # Sum upper triangle to avoid double counting and self-overlap
                    total_overlap_loss_sum += torch.triu(iou_mat, diagonal=1).sum()

            # --- Containment Loss (per sample) ---
            # Dialogs
            current_dialog_mask = dialog_mask_batch[b]
            if (current_dialog_mask.any() and dialog_bboxes_cxywh_batch is not None and
                dialog_breakout_ratio_batch is not None and dialog_parent_idx_batch is not None and
                panel_bboxes_cxywh_batch is not None):

                child_bboxes_cxywh_sample = dialog_bboxes_cxywh_batch[b][current_dialog_mask]
                breakout_ratio_sample = dialog_breakout_ratio_batch[b][current_dialog_mask]
                parent_idx_sample = dialog_parent_idx_batch[b][current_dialog_mask]

                if child_bboxes_cxywh_sample.numel() > 0:
                    child_xyxy_sample = _cxywh_to_xyxy(child_bboxes_cxywh_sample)
                    # Select parent panel bboxes using parent_idx_sample
                    parent_panel_bboxes_cxywh_sample = panel_bboxes_cxywh_batch[b][parent_idx_sample]
                    parent_xyxy_sample = _cxywh_to_xyxy(parent_panel_bboxes_cxywh_sample)
                    total_containment_loss_sum += self._containment_loss(
                        child_xyxy_sample, parent_xyxy_sample, breakout_ratio_sample
                    )

            # Characters
            current_character_mask = character_mask_batch[b]
            if (current_character_mask.any() and char_bboxes_cxywh_batch is not None and
                char_breakout_ratio_batch is not None and char_parent_idx_batch is not None and
                panel_bboxes_cxywh_batch is not None):

                child_bboxes_cxywh_sample = char_bboxes_cxywh_batch[b][current_character_mask]
                breakout_ratio_sample = char_breakout_ratio_batch[b][current_character_mask]
                parent_idx_sample = char_parent_idx_batch[b][current_character_mask]

                if child_bboxes_cxywh_sample.numel() > 0:
                    child_xyxy_sample = _cxywh_to_xyxy(child_bboxes_cxywh_sample)
                    # Select parent panel bboxes using parent_idx_sample
                    parent_panel_bboxes_cxywh_sample = panel_bboxes_cxywh_batch[b][parent_idx_sample]
                    parent_xyxy_sample = _cxywh_to_xyxy(parent_panel_bboxes_cxywh_sample)
                    total_containment_loss_sum += self._containment_loss(
                        child_xyxy_sample, parent_xyxy_sample, breakout_ratio_sample
                    )

        # Average over batch size
        num_samples = B
        overlap_loss = total_overlap_loss_sum / max(1, num_samples)
        containment_loss = total_containment_loss_sum / max(1, num_samples)
        
        # The total loss for this component
        total_geom_loss = overlap_loss + containment_loss

        loss_dict = {
            "geom_overlap_loss": overlap_loss.item(),
            "geom_containment_loss": containment_loss.item()
        }
        return total_geom_loss, loss_dict

class LayoutCompositeLoss(nn.Module): # Renamed back to original name for consistency
    """
    Combines Prediction Loss, Style Loss, and Geometric Constraint Loss.
    """
    def __init__(self, lambda_style=0.1, lambda_geom=0.1):
        super().__init__()
        self.lambda_style = float(lambda_style)
        self.lambda_geom = float(lambda_geom)
        self.pred_loss_fn = PredictionLoss()
        self.style_calc_fn = StyleCalculator()
        self.geom_loss_fn = GeometricConstraintLoss()
        self.mse = nn.MSELoss(reduction='mean') # For style loss comparison

    def forward(self, predictions, batch):
        """
        Calculates the total composite loss.
        predictions: A dictionary of batched tensors (B, N, D) or (B, N).
                     e.g., {'panel_bbox': (B, Pmax, 4), 'panel_offsets': (B, Pmax, 8),
                            'panel_class_logits': (B, Pmax, C), ...}
        batch: A dictionary of batched tensors (B, N, D) or (B, N).
               e.g., {'panel_bboxes': (B, Pmax, 4), 'panel_classes': (B, Pmax),
                      'panel_mask': (B, Pmax), 'style_vector': (B, 4),
                      'dialog_parent_idx': (B, Dmax), 'char_parent_idx': (B, Cmax), ...}
        """
        # Determine device from an existing tensor in batch or predictions
        device = None
        if 'panel_mask' in batch and batch['panel_mask'] is not None:
            device = batch['panel_mask'].device
        elif 'panel_bbox' in predictions and predictions['panel_bbox'] is not None:
            device = predictions['panel_bbox'].device
        else:
            device = torch.device("cpu") # Fallback

        # 1. Prediction Loss (Geometric properties and classifications)
        # This now returns a single scalar loss and a dict of components
        pred_loss, pred_loss_dict = self.pred_loss_fn(predictions, batch)

        # 2. Style Loss
        # StyleCalculator now returns the (B, 4) predicted style values
        style_pred_values = self.style_calc_fn(predictions, batch) # (B, 4) tensor
        
        # Calculate style loss using MSE with ground truth style vector
        style_gt = batch["style_vector"].to(device) # (B, 4) ground truth style vector
        style_loss = self.mse(style_pred_values, style_gt)
        style_loss_dict = {"style_loss": style_loss.item()}

        # 3. Geometric Constraint Loss (Overlap and Containment)
        # This also returns a single scalar loss and a dict of components
        geom_loss, geom_loss_dict = self.geom_loss_fn(predictions, batch)

        # Combine all losses
        total_loss = pred_loss + self.lambda_style * style_loss + self.lambda_geom * geom_loss

        # Aggregate all loss items for logging
        loss_dict = {
            **pred_loss_dict,
            **style_loss_dict,
            **geom_loss_dict,
            "geom_loss": geom_loss.item(),
            "pred_loss": pred_loss.item(),
            "style_loss": style_loss.item(),
            "total_loss": total_loss.item()
        }
        return total_loss, loss_dict

