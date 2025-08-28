# DiffSensei-main/layout-generator/models/layout_planner/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PanelClassHead(nn.Module):
    """Panel Shape Type Prediction Head"""
    def __init__(self, d_model, num_classes=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)  # 输出 logits
        )
    
    def forward(self, x):
        return self.mlp(x)  # Shape: (num_panels, num_classes)

class PanelBBoxHead(nn.Module):
    """Panel BBox Prediction Head"""
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4),  # 输出 (x_center, y_center, width, height)
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
    
    def forward(self, x):
        return self.mlp(x)  # Shape: (num_panels, 4)

class PanelOffsetsHead(nn.Module):
    """Panel Offsets Prediction Head"""
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 8)  # 输出 8 个 offsets
        )
    
    def forward(self, x):
        return self.mlp(x)  # Shape: (num_panels, 8)

class ElementBBoxHead(nn.Module):
    """Dialog/Character BBox Prediction Head"""
    def __init__(self, d_model):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 4),  # 输出 (x_center, y_center, width, height)
        #     nn.Sigmoid()  # 归一化到 [0, 1]
        # )
        # 输入维度加倍，因为拼接了父panel特征
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )
    
    # def forward(self, x):
    #     return self.mlp(x)  # Shape: (num_elements, 4)
    def forward(self, x, parent_panel_features):
        if parent_panel_features is None:
            parent_panel_features = torch.zeros_like(x)
        fused = torch.cat([x, parent_panel_features], dim=-1)
        return self.mlp(fused)

class BreakoutHead(nn.Module):
    """Breakout Prediction Head (Classification + Ratio)"""
    def __init__(self, d_model):
        super().__init__()
        self.mlp_class = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),  # 融合父 Panel 特征
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)  # 输出 breakout 分类 logit
        )
        self.mlp_ratio = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),  # 输出 breakout ratio
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
    
    # def forward(self, x, parent_panel_features):
    #     # 融合当前元素和父 Panel 特征
    #     fused_features = torch.cat([x, parent_panel_features], dim=-1)
    #     breakout_logit = self.mlp_class(fused_features)  # Shape: (num_elements, 1)
    #     breakout_ratio = self.mlp_ratio(fused_features)  # Shape: (num_elements, 1)
    #     return breakout_logit, breakout_ratio
    def forward(self, x, parent_panel_features):
        if parent_panel_features is None:
            # fallback: use zeros for parent
            parent_panel_features = torch.zeros_like(x)
        fused = torch.cat([x, parent_panel_features], dim=-1)
        return self.mlp_class(fused), self.mlp_ratio(fused)

class DialogShapeHead(nn.Module):
    """Dialog Bubble Shape Prediction Head"""
    def __init__(self, d_model, num_shapes=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_shapes)  # 输出 bubble shape logits
        )
    
    def forward(self, x):
        return self.mlp(x)  # Shape: (num_dialogs, num_shapes)

    
class ParallelPredictionHeads(nn.Module):
    def __init__(self, d_model=512, num_panel_classes=4, num_dialog_shapes=4,
                 layout_types=None, max_panels=30, max_dialogs=30, max_chars=30):
        super().__init__()
        # Panel Prediction Heads
        self.panel_class_head = PanelClassHead(d_model, num_panel_classes)
        self.panel_bbox_head = PanelBBoxHead(d_model)
        self.panel_offsets_head = PanelOffsetsHead(d_model)

        # Dialog/Character Prediction Heads
        self.element_bbox_head = ElementBBoxHead(d_model)
        self.breakout_head = BreakoutHead(d_model)
        self.dialog_shape_head = DialogShapeHead(d_model, num_dialog_shapes)
        self.layout_types = layout_types or {}

        self.max_panels = max_panels
        self.max_dialogs = max_dialogs
        self.max_chars = max_chars

    def forward(self, lfm_output, element_types, element_indices, parent_panel_indices):
        B, S, D = lfm_output.shape
        device = lfm_output.device

        # mask
        panel_mask = (element_types == self.layout_types['TYPE_PANEL'])
        char_mask = (element_types == self.layout_types['TYPE_CHAR'])
        dialog_mask = (element_types == self.layout_types['TYPE_DIALOG'])

        panel_features = lfm_output[panel_mask]
        dialog_features = lfm_output[dialog_mask]
        character_features = lfm_output[char_mask]

        outputs = {}

        # ======== Panel Predictions (pad to max_panels) ========
        if panel_features.numel() > 0:
            raw_cls_logits = self.panel_class_head(panel_features)
            raw_bbox = self.panel_bbox_head(panel_features)
            raw_offsets = self.panel_offsets_head(panel_features)

            cls_logits_padded = torch.zeros((B, self.max_panels, raw_cls_logits.shape[-1]), device=device)
            bbox_padded = torch.zeros((B, self.max_panels, raw_bbox.shape[-1]), device=device)
            offsets_padded = torch.zeros((B, self.max_panels, raw_offsets.shape[-1]), device=device)

            start = 0
            for b in range(B):
                num_p = panel_mask[b].sum().item()
                fill_n = min(num_p, self.max_panels)
                if fill_n > 0:
                    cls_logits_padded[b, :fill_n] = raw_cls_logits[start:start+fill_n]
                    bbox_padded[b, :fill_n] = raw_bbox[start:start+fill_n]
                    offsets_padded[b, :fill_n] = raw_offsets[start:start+fill_n]
                start += num_p

            outputs['panel_class_logits'] = cls_logits_padded
            outputs['panel_bbox'] = bbox_padded
            outputs['panel_offsets'] = offsets_padded

        # ======== Dialog & Character Predictions (pad to max_dialogs/chars) ========
        if dialog_features.numel() > 0 or character_features.numel() > 0:
            batch_idx_panels = torch.nonzero(panel_mask, as_tuple=True)[0]
            dialog_parent_feats = self._get_parent_features(
                dialog_mask, parent_panel_indices, panel_mask, element_indices, batch_idx_panels, panel_features
            )
            char_parent_feats = self._get_parent_features(
                char_mask, parent_panel_indices, panel_mask, element_indices, batch_idx_panels, panel_features
            )

            # --- Dialog ---
            if dialog_features.numel() > 0:
                # raw_dialog_bbox = self.element_bbox_head(dialog_features)
                raw_dialog_bbox = self.element_bbox_head(dialog_features, dialog_parent_feats) # <-- 传入父特征

                bl, br = self.breakout_head(dialog_features, dialog_parent_feats)
                raw_shape_logits = self.dialog_shape_head(dialog_features)

                bbox_pad = torch.zeros((B, self.max_dialogs, 4), device=device)
                bl_pad = torch.zeros((B, self.max_dialogs, 1), device=device)
                br_pad = torch.zeros((B, self.max_dialogs, 1), device=device)
                shape_pad = torch.zeros((B, self.max_dialogs, raw_shape_logits.shape[-1]), device=device)

                start = 0
                for b in range(B):
                    num_d = dialog_mask[b].sum().item()
                    fill_n = min(num_d, self.max_dialogs)
                    if fill_n > 0:
                        bbox_pad[b, :fill_n] = raw_dialog_bbox[start:start+fill_n]
                        bl_pad[b, :fill_n] = bl[start:start+fill_n]
                        br_pad[b, :fill_n] = br[start:start+fill_n]
                        shape_pad[b, :fill_n] = raw_shape_logits[start:start+fill_n]
                    start += num_d

                outputs['dialog_bbox'] = bbox_pad
                outputs['dialog_breakout_logits'] = bl_pad
                outputs['dialog_breakout_ratio'] = br_pad
                outputs['dialog_shape_logits'] = shape_pad

            # --- Character ---
            if character_features.numel() > 0:
                # raw_char_bbox = self.element_bbox_head(character_features)
                raw_char_bbox = self.element_bbox_head(character_features, char_parent_feats) # <-- 传入父特征

                bl, br = self.breakout_head(character_features, char_parent_feats)

                bbox_pad = torch.zeros((B, self.max_chars, 4), device=device)
                bl_pad = torch.zeros((B, self.max_chars, 1), device=device)
                br_pad = torch.zeros((B, self.max_chars, 1), device=device)

                start = 0
                for b in range(B):
                    num_c = char_mask[b].sum().item()
                    fill_n = min(num_c, self.max_chars)
                    if fill_n > 0:
                        bbox_pad[b, :fill_n] = raw_char_bbox[start:start+fill_n]
                        bl_pad[b, :fill_n] = bl[start:start+fill_n]
                        br_pad[b, :fill_n] = br[start:start+fill_n]
                    start += num_c

                outputs['character_bbox'] = bbox_pad
                outputs['character_breakout_logits'] = bl_pad
                outputs['character_breakout_ratio'] = br_pad

        return outputs

    def _get_parent_features(self, child_mask, parent_indices_all, panel_mask, panel_indices_all, batch_idx_panels, panel_features):
        """ 高效查找父 Panel 特征的辅助函数 """
        if torch.count_nonzero(child_mask) == 0:
            return None
        
        device = panel_features.device
        B, S = child_mask.shape
        
        # a. 获取子元素对应的批次索引
        batch_idx_child = torch.nonzero(child_mask, as_tuple=True)[0]
        
        # b. 获取子元素的父 Panel 原始索引
        parent_indices_child = parent_indices_all[child_mask] # (TotalChildren,)
        
        # c. 构建一个映射：(batch_idx, panel_orig_idx) -> panel_compressed_idx
        # panel_indices_all[panel_mask] -> (TotalPanels,) 得到所有 panel 的原始索引
        # 这三行代码创建了一个查找表
        map_keys = torch.stack([batch_idx_panels, panel_indices_all[panel_mask]], dim=1)
        map_values = torch.arange(panel_features.shape[0], device=device)
        
        # d. 准备查询键：(batch_idx_child, parent_indices_child)
        query_keys = torch.stack([batch_idx_child, parent_indices_child], dim=1)

        # e. 执行查找 (这是一个简化的哈希查找，实际需要更鲁棒的实现)
        # 我们使用一种更直接的广播和比较方法
        # 找到每个 child 的 parent 在 panel_features 中的索引
        # (TotalChildren, 1) == (1, TotalPanels) -> (TotalChildren, TotalPanels)
        match_matrix = (query_keys[:, 0:1] == map_keys[:, 0:1].T) & \
                       (query_keys[:, 1:2] == map_keys[:, 1:2].T)
        
        # 找到匹配位置
        found_indices = torch.nonzero(match_matrix, as_tuple=True)[1]
        
        # f. 处理未找到父Panel的情况 (例如父Panel被padding截断)
        # 创建一个默认的零特征张量
        parent_features = torch.zeros(child_mask.sum(), panel_features.shape[1], device=device)
        
        # 仅对找到了父Panel的子元素，用 gather 提取特征
        valid_child_mask = (parent_indices_child >= 0)
        if valid_child_mask.any():
            parent_features[valid_child_mask] = panel_features[found_indices]
            
        return parent_features