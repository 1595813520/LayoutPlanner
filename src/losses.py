import einops
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
        self.ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean')

    def masked_loss(self, pred, target, mask, loss_fn, reshape_logits=False):
        """
        Generic masked loss calculator.
        pred: (B, N, D) or (B, N)
        target: same shape except last dim for classification logits
        mask: (B, N) bool
        loss_fn: loss object with signature loss_fn(pred, target)
        reshape_logits: for CE loss that expects 2D logits (Ntot, C) and 1D target
        """
        if pred is None or target is None or mask is None:
            # return torch.tensor(0.0, device=pred.device if pred is not None else torch.device("cuda"), requires_grad=True)
            return (pred*0).sum() if pred is not None else torch.tensor(0.0, device=mask.device)
        
        if not mask.any():
            # return torch.tensor(0.0, device=pred.device, requires_grad=True)  # 保证梯度链不断
            return (pred*0).sum()
        
        if reshape_logits:
            pred_masked = pred[mask]
            target_masked = target[mask]
            if pred_masked.numel() == 0:
                # return None
                # return torch.tensor(0.0, device=pred.device, requires_grad=True)
                return (pred*0).sum()
            
            # return loss_fn(pred_masked.view(-1, pred_masked.shape[-1]), target_masked.view(-1))
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss) or loss_fn == F.cross_entropy:
                return loss_fn(pred_masked.view(-1, pred_masked.shape[-1]), target_masked.view(-1).long())
            else:
                return loss_fn(pred_masked, target_masked.float())
        else:
            pred_masked = pred[mask]
            target_masked = target[mask]
            if pred_masked.ndim == 2 and pred_masked.shape[1] == 1:
                pred_masked = pred_masked.squeeze(-1)
            if target_masked.ndim == 2 and target_masked.shape[1] == 1:
                target_masked = target_masked.squeeze(-1)
            if pred_masked.numel() == 0:
                return None
            pred_masked = torch.nan_to_num(pred_masked, nan=0.0, posinf=0.0, neginf=0.0)
            target_masked = torch.nan_to_num(target_masked, nan=0.0, posinf=0.0, neginf=0.0)
            # 新增nan屏蔽
            if torch.isnan(pred_masked).any() or torch.isnan(target_masked).any():
                # return None
                raise ValueError('NaN detected!')
            return loss_fn(pred_masked, target_masked.float())

    def forward(self, predictions, batch):
        device = batch['panel_mask'].device if 'panel_mask' in batch else torch.device("cpu")
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # Mapping: pred_key, target_key, mask_key, loss_fn, reshape_logits
        loss_items = [
            # Panels
            ('panel_class_logits', 'panel_classes', 'panel_mask', self.ce, True),
            ('panel_bbox', 'panel_bboxes', 'panel_mask', self.l1, False),
            ('panel_offsets', 'panel_offsets', 'panel_mask', self.l1, False),
            # Dialogs
            ('dialog_bbox', 'dialog_bboxes', 'dialog_mask', self.l1, False),
            ('dialog_breakout_logits', 'dialog_breakout_labels', 'dialog_mask', self.bce, False),
            ('dialog_breakout_ratio', 'dialog_breakout_ratios', 'dialog_mask', self.l1, False),
            ('dialog_shape_logits', 'dialog_shapes', 'dialog_mask', self.ce, True),
            # Characters
            ('char_bbox', 'character_bboxes', 'character_mask', self.l1, False),
            ('character_breakout_logits', 'character_breakout_labels', 'character_mask', self.bce, False),
            ('character_breakout_ratio', 'character_breakout_ratios', 'character_mask', self.l1, False),
        ]

        for pred_k, target_k, mask_k, fn, reshape in loss_items:
            pred_tensor = predictions.get(pred_k)
            target_tensor = batch.get(target_k)
            mask_tensor = batch.get(mask_k)
            loss_val = self.masked_loss(pred_tensor, target_tensor, mask_tensor, fn, reshape_logits=reshape)
            if loss_val is not None:
                total_loss += loss_val
                loss_name = pred_k.replace('_logits','').replace('_bbox','') \
                                  .replace('_ratio','').replace('_classes','') \
                                  .replace('_labels','') \
                                  .replace('_shapes','') \
                                  .replace('_offsets','')
                loss_dict[loss_name] = loss_val.item()

        return total_loss, loss_dict
    
def nanvar(x, dim=None, keepdim=False):
    mask = ~torch.isnan(x)
    x2 = torch.where(mask, x, torch.zeros_like(x))
    count = mask.sum(dim=dim, keepdim=keepdim)
    mean = x2.sum(dim=dim, keepdim=keepdim) / count.clamp(min=1)
    if not keepdim and dim is not None:
        mean = mean.unsqueeze(dim)
    var = (((torch.where(mask, x, mean) - mean) ** 2).sum(dim=dim, keepdim=keepdim)) / count.clamp(min=1)
    var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    return var

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

        layout_density = torch.nan_to_num(p_area.sum(dim=1) / hull_area, nan=0.0)  # (B,)

        # 2) alignment_score (mask 少于2个时给0.5)
        valid_counts = panel_mask.sum(dim=1)
        cx, cy = self._cxcy_from_xyxy01(p_bbox_xyxy)
        cx = torch.where(panel_mask, cx, torch.nan)
        cy = torch.where(panel_mask, cy, torch.nan)
        var_x = torch.nan_to_num(nanvar(cx, dim=1), nan=0.0)
        var_y = torch.nan_to_num(nanvar(cy, dim=1), nan=0.0)
        alignment_score = torch.where(valid_counts > 1, 1.0 / (1.0 + var_x + var_y + self.eps),
                                      torch.full_like(var_x, 0.5)).nan_to_num(0.5)

        # 3) shape_instability
        rms_per_panel = torch.sqrt((p_offsets ** 2).mean(dim=-1).clamp(min=0.0) + self.eps)
        rms_per_panel = torch.where(panel_mask, rms_per_panel, torch.nan)
        shape_instability = torch.nan_to_num(torch.nanmean(rms_per_panel, dim=1), nan=0.0)

        # 4) breakout_intensity (批处理)
        def process_breakout(bbox_tensor, ratio_tensor, mask_tensor):
            xyxy = _cxywh_to_xyxy(bbox_tensor)
            area = self._area_xyxy01(xyxy)  # (B, N)
            ratio = ratio_tensor.squeeze(-1) if ratio_tensor.ndim > 2 else ratio_tensor
            area = torch.where(mask_tensor, area, 0.0)
            ratio = torch.where(mask_tensor, ratio, 0.0)
            return area, ratio

        # print("predictions keys:", predictions.keys())
        
        d_area, d_ratio = process_breakout(predictions['dialog_bbox'],
                                           predictions['dialog_breakout_ratio'],
                                           batch['dialog_mask'])
        c_area, c_ratio = process_breakout(predictions['character_bbox'],
                                           predictions['character_breakout_ratio'],
                                           batch['character_mask'])
        all_area = torch.cat([d_area, c_area], dim=1)  # (B, Dmax+Cmax)
        all_ratio = torch.cat([d_ratio, c_ratio], dim=1)
        denom = all_area.sum(dim=1).clamp(min=self.eps)
        breakout_intensity = torch.where(
            denom > self.eps,
            (all_area * all_ratio).sum(dim=1) / denom,
            torch.where(all_ratio.sum(dim=1) > 0, all_ratio.mean(dim=1), torch.zeros_like(denom))
        ).nan_to_num(0.0)

        # Stack batch style vector
        style_pred = torch.stack([
            layout_density,
            alignment_score,
            shape_instability,
            breakout_intensity
        ], dim=-1)
        return style_pred
    

def xywh_2_ltrb(bbox_xywh):

    bbox_ltrb = torch.zeros(bbox_xywh.shape).to(bbox_xywh.device)
    bbox_xy = torch.abs(bbox_xywh[:, :, :2])
    bbox_wh = torch.abs(bbox_xywh[:, :, 2:])
    bbox_ltrb[:, :, :2] = bbox_xy - 0.5 * bbox_wh
    bbox_ltrb[:, :, 2:] = bbox_xy + 0.5 * bbox_wh
    return bbox_ltrb


def ltrb_2_xywh(bbox_ltrb):
    bbox_xywh = torch.zeros(bbox_ltrb.shape)
    bbox_wh = torch.abs(bbox_ltrb[:, :, 2:] - bbox_ltrb[:, :, :2])
    bbox_xy = bbox_ltrb[:, :, :2] + 0.5 * bbox_wh
    bbox_xywh[:, :, :2] = bbox_xy
    bbox_xywh[:, :, 2:] = bbox_wh
    return bbox_xywh


def xywh_to_ltrb_split(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def rand_bbox_ltrb(batch_shape):

    bbox_lt = torch.rand(batch_shape + [2])
    bbox_wh_max = 1 - bbox_lt
    bbox_wh_weight = torch.rand(batch_shape).unsqueeze(-1).repeat([1 for _ in range(len(batch_shape))] + [2])

    bbox_wh = 1 * bbox_wh_weight * bbox_wh_max
    bbox_rb = bbox_lt + bbox_wh

    bbox = torch.cat([bbox_lt, bbox_rb], dim=-1)
    return bbox


def rand_bbox_xywh(batch_shape):

    bbox_ltrb = rand_bbox_ltrb(batch_shape)
    bbox_xywh = ltrb_2_xywh(bbox_ltrb)
    return bbox_xywh


def GIoU_ltrb(bbox_1, bbox_2):

    # step 1 calculate area of bbox_1 and bbox_2
    a_1 = (bbox_1[:, :, 2] - bbox_1[:, :, 0]) * (bbox_1[:, :, 3] - bbox_1[:, :, 1])
    a_2 = (bbox_2[:, :, 2] - bbox_2[:, :, 0]) * (bbox_2[:, :, 3] - bbox_2[:, :, 1])

    # step 2.1 compute intersection I bbox
    bbox = torch.cat([bbox_1.unsqueeze(-1), bbox_2.unsqueeze(-1)], dim=-1)
    bbox_I_lt = torch.max(bbox, dim=-1)[0][:, :, :2]
    bbox_I_rb = torch.min(bbox, dim=-1)[0][:, :, 2:]

    # step 2.2 compute area of I
    a_I = F.relu((bbox_I_rb[:, :, 0] - bbox_I_lt[:, :, 0])) * F.relu((bbox_I_rb[:, :, 1] - bbox_I_lt[:, :, 1]))

    # step 3.1 compute smallest enclosing box C
    bbox_C_lt = torch.min(bbox, dim=-1)[0][:, :, :2]
    bbox_C_rb = torch.max(bbox, dim=-1)[0][:, :, 2:]

    # step 3.2 compute area of C
    a_C = (bbox_C_rb[:, :, 0] - bbox_C_lt[:, :, 0]) * (bbox_C_rb[:, :, 1] - bbox_C_lt[:, :, 1])

    # step 4 compute IoU
    a_U = (a_1 + a_2 - a_I)
    iou = a_I / (a_U + 1e-10)

    # step 5 copute giou
    giou = iou - (a_C - a_U) / (a_C + 1e-10)

    return iou, giou


def GIoU_xywh(bbox_pred, bbox_true, xy_only=False):

    if xy_only:
        wh = torch.abs(bbox_pred[:, :, 2:].clone().detach())
        bbox = torch.cat([bbox_pred[:, :, :2], wh], dim=2)
    else:
        bbox = bbox_pred

    bbox_pred_ltrb = xywh_2_ltrb(torch.abs(bbox))
    bbox_true_ltrb = xywh_2_ltrb(torch.abs(bbox_true))
    return GIoU_ltrb(bbox_pred_ltrb, bbox_true_ltrb)


def PIoU_ltrb(bbox_ltrb, mask=None):

    n_box = bbox_ltrb.shape[1]
    device = bbox_ltrb.device

    # compute area of bboxes
    area_bbox = (bbox_ltrb[:, :, 2] - bbox_ltrb[:, :, 0]) * (bbox_ltrb[:, :, 3] - bbox_ltrb[:, :, 1])
    area_bbox_psum = area_bbox.unsqueeze(-1) + area_bbox.unsqueeze(-2)

    # compute pairwise intersection
    x1y1 = bbox_ltrb[:, :, [0, 1]]
    x1y1 = torch.swapaxes(x1y1, 1, 2)
    x1y1_I = torch.max(x1y1.unsqueeze(-1), x1y1.unsqueeze(-2))

    x2y2 = bbox_ltrb[:, :, [2, 3]]
    x2y2 = torch.swapaxes(x2y2, 1, 2)
    x2y2_I = torch.min(x2y2.unsqueeze(-1), x2y2.unsqueeze(-2))
    # compute area of Is
    wh_I = F.relu(x2y2_I - x1y1_I)
    area_I = wh_I[:, 0, :, :] * wh_I[:, 1, :, :]

    # compute pairwise IoU
    piou = area_I / (area_bbox_psum - area_I + 1e-10)

    piou.masked_fill_(torch.eye(n_box, n_box).to(torch.bool).to(device), 0)

    if mask is not None:
        mask = mask.unsqueeze(2)
        select_mask = torch.matmul(mask, torch.transpose(mask, dim0=1, dim1=2))
        piou = piou * select_mask.to(device)

    return piou

def PIoU_xywh(bbox_xywh, mask=None, xy_only=True):

    if xy_only:
        wh = torch.abs(bbox_xywh[:, :, 2:].clone().detach())
        bbox = torch.cat([bbox_xywh[:, :, :2], wh], dim=2)
        bbox_ltrb = xywh_2_ltrb(bbox)
    else:
        bbox_ltrb = xywh_2_ltrb(bbox_xywh)

    return PIoU_ltrb(bbox_ltrb, mask)


def Pdist(bbox):
    xy = bbox[:, :, :2]
    pdist_m = torch.cdist(xy, xy, p=2)

    return pdist_m

def layout_alignment(bbox, mask, xy_only=False, mode='all'):
    """
    alignment metrics in Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    """

    if xy_only:
        wh = torch.abs(bbox[:, :, 2:].clone().detach())
        bbox = torch.cat([bbox[:, :, :2], wh], dim=2)

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = xywh_to_ltrb_split(bbox)
    xc, yc = bbox[0], bbox[1]
    if mode == 'all':
        X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    elif mode == 'partial':
        X = torch.stack([xl, xc, yt, yb], dim=1)
    else:
        raise Exception('mode must be all or partial')

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0

    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)
    X = -torch.log(1 - X)

    score = einops.reduce(X, "b s -> b", reduction="sum")
    score_normalized = score / einops.reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    return score, score_normalized


def layout_alignment_matrix(bbox, mask):
    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = xywh_to_ltrb_split(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    return X

class GeometricConstraintLoss(nn.Module):
    def __init__(self, breakout_tol_min=0.01, breakout_tol_max=0.3, rect_class_id=0, eps=1e-6, align_weight=1.0, overlap_weight=1.0):
        super().__init__()
        self.breakout_tol_min = breakout_tol_min
        self.breakout_tol_max = breakout_tol_max
        self.eps = eps
        self.align_weight = align_weight
        self.overlap_weight = overlap_weight
        self.rect_class_id = rect_class_id

    def _containment_loss_batch(self, child_xyxy, parent_xyxy, valid_mask, tol=0.0):
        lt = torch.max(child_xyxy[..., :2], parent_xyxy[..., :2])
        rb = torch.min(child_xyxy[..., 2:], parent_xyxy[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        child_area = (child_xyxy[..., 2] - child_xyxy[..., 0]) * (child_xyxy[..., 3] - child_xyxy[..., 1]).clamp(min=self.eps)
        outside_area = (child_area - inter).clamp(min=0)
        # 平滑容忍机制
        loss_per_element = F.relu(outside_area / (child_area + self.eps) - tol)
        
        mask = valid_mask  # 不固定 breakout_thresh，动态 tol 控制
        return loss_per_element[mask].mean() if mask.any() else torch.tensor(
            0.0, device=child_xyxy.device, requires_grad=True)

    def forward(self, predictions, batch, style_target_breakout=None):
        device = batch['panel_mask'].device
        B = batch['panel_mask'].shape[0]

        # --- 取 panel bboxes ---
        panel_mask = batch['panel_mask']  # (B, Pmax)
        panel_bboxes_xywh = predictions['panel_bbox']  # 这里假设是 [0,1] 范围
        panel_bboxes_xyxy = _cxywh_to_xyxy(panel_bboxes_xywh)

        # --- Overlap: LACE风格 ---
        piou = PIoU_xywh(panel_bboxes_xywh, mask=panel_mask.to(torch.float32), xy_only=False)  # (B, P, P)
        piou = torch.nan_to_num(piou, nan=0.0, posinf=0.0, neginf=0.0)
        pdist = Pdist(panel_bboxes_xywh)  # (B, P, P)
        overlap_term = torch.mean(piou, dim=[1, 2]) + torch.mean(piou.ne(0) * torch.exp(-pdist), dim=[1, 2])
        overlap_loss = overlap_term.mean()

        # --- Alignment: LACE全局对齐 ---
        _, align_loss = layout_alignment(panel_bboxes_xywh, mask=panel_mask, xy_only=False)
        align_loss = align_loss.mean()

        
        # ========== 动态 tol ==========
        if style_target_breakout is not None:
            tol_value = self.breakout_tol_min + \
                        (self.breakout_tol_max - self.breakout_tol_min) * style_target_breakout
        else:
            tol_value = torch.full((B,), self.breakout_tol_min, device=device)
            
        # Containment: Dialog
        dialog_mask = batch['dialog_mask']
        dialog_parent_idx = batch['dialog_parent_idx'].clamp(0, panel_bboxes_xyxy.shape[1] - 1)
        dialog_parent_boxes = torch.gather(panel_bboxes_xyxy, 1,
                                           dialog_parent_idx.unsqueeze(-1).expand(-1, -1, 4))
        dialog_child_bbox = _cxywh_to_xyxy(predictions['dialog_bbox'])
        dialog_ratio = predictions['dialog_breakout_ratio'].squeeze(-1) \
            if predictions['dialog_breakout_ratio'].dim() > 2 \
            else predictions['dialog_breakout_ratio']
        dialog_loss_batch = []
        for b in range(B):
            dialog_loss_batch.append(
                self._containment_loss_batch(dialog_child_bbox[b:b+1],
                                             dialog_parent_boxes[b:b+1],
                                             dialog_ratio[b:b+1],
                                             dialog_mask[b:b+1],
                                             tol=tol_value[b].item()))
        dialog_loss = torch.stack(dialog_loss_batch).mean()

        # Containment: Char
        char_mask = batch['character_mask']
        char_parent_idx = batch['char_parent_idx'].clamp(0, panel_bboxes_xyxy.shape[1] - 1)
        char_parent_boxes = torch.gather(panel_bboxes_xyxy, 1,
                                         char_parent_idx.unsqueeze(-1).expand(-1, -1, 4))
        char_child_bbox = _cxywh_to_xyxy(predictions['character_bbox'])
        char_ratio = predictions['character_breakout_ratio'].squeeze(-1) \
            if predictions['character_breakout_ratio'].dim() > 2 \
            else predictions['character_breakout_ratio']
        char_loss_batch = []
        for b in range(B):
            char_loss_batch.append(
                self._containment_loss_batch(char_child_bbox[b:b+1],
                                             char_parent_boxes[b:b+1],
                                             char_ratio[b:b+1],
                                             char_mask[b:b+1],
                                             tol=tol_value[b].item()))
        char_loss = torch.stack(char_loss_batch).mean()

        containment_loss = dialog_loss + char_loss
        
        # 逻辑约束：矩形 offset=0
        if 'panel_offsets' in predictions:
            rect_mask = (batch['panel_classes'] == self.rect_class_id) & batch['panel_mask']
            if rect_mask.any():
                rect_offsets = predictions['panel_offsets'][rect_mask]
                logic_rect_offset_loss = F.l1_loss(rect_offsets, torch.zeros_like(rect_offsets))
            else:
                logic_rect_offset_loss = torch.tensor(0.0, device=device)
        else:
            logic_rect_offset_loss = torch.tensor(0.0, device=device)
            
        # --- 总几何loss ---
        total_geom_loss = (
            self.overlap_weight * overlap_loss +
            self.align_weight * align_loss +
            containment_loss + 
            logic_rect_offset_loss
        )
        
        loss_dict = {
            "geom_overlap_loss": overlap_loss.item(),
            "geom_align_loss": align_loss.item(),
            "geom_containment_loss": containment_loss.item(),
            "geom_logic_rect_offset_loss": logic_rect_offset_loss.item(),
        }

        return total_geom_loss, loss_dict
    

class LayoutCompositeLoss(nn.Module):
    def __init__(self, lambda_style=20.0, lambda_geom=20.0, style_mu=None, style_sigma=None, rect_class_id=0):
        super().__init__()
        self.lambda_style = lambda_style
        self.lambda_geom = lambda_geom
        self.pred_loss_fn = PredictionLoss()            # 已经批处理
        self.style_calc_fn = StyleCalculator()          # 已向量化
        self.geom_loss_fn = GeometricConstraintLoss()   # 新批处理版
        self.mse = nn.MSELoss(reduction='mean')
        self.rect_class_id = rect_class_id
        self.register_buffer('style_mu', torch.tensor(style_mu, dtype=torch.float32))
        self.register_buffer('style_sigma', torch.tensor(style_sigma, dtype=torch.float32))

    def forward(self, predictions, batch):
        device = batch['panel_mask'].device

        # PredictionLoss
        pred_loss, pred_loss_dict = self.pred_loss_fn(predictions, batch)

        # StyleLoss
        style_pred_values = self.style_calc_fn(predictions, batch)  # (B, 4)
        style_gt = batch["style_vector"].to(device)                 # (B, 4)
        
        # ===== 标准化 =====
        style_pred_norm = (style_pred_values - self.style_mu.to(device)) / self.style_sigma.to(device)
        style_gt_norm = (style_gt - self.style_mu.to(device)) / self.style_sigma.to(device)
        style_loss = self.mse(style_pred_norm, style_gt_norm)
        # style_loss = self.mse(style_pred_values, style_gt)

        # Geometric Loss
        # geom_loss, geom_loss_dict = self.geom_loss_fn(predictions, batch)
        
        # geom_loss 接收 style_target_breakout（未标准化的第4列，用来计算 tol）
        style_target_breakout = style_gt[:, 3]
        geom_loss, geom_loss_dict = self.geom_loss_fn(predictions, batch,
                                                      style_target_breakout=style_target_breakout, rect_class_id=self.rect_class_id)

        # pred_loss  = torch.nan_to_num(pred_loss, nan=0.0)
        # style_loss = torch.nan_to_num(style_loss, nan=0.0)
        # geom_loss  = torch.nan_to_num(geom_loss, nan=0.0)

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
