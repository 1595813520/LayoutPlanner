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
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4),  # 输出 (x_center, y_center, width, height)
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
    
    def forward(self, x):
        return self.mlp(x)  # Shape: (num_elements, 4)

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
    def __init__(self, d_model=512, num_panel_classes=4, num_dialog_shapes=4, layout_types=None):
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
        
    
    def forward(self, lfm_output, element_types, element_indices, parent_panel_indices):
        """
        lfm_output: (B, S, D) - LFM 输出特征序列
        element_types: (B, S) - 每个 token 的类型
        element_indices: (B, S) - 每个 token 的自身索引 (0, 1, 2...)
        parent_panel_indices: (B, S) - 每个元素的父 Panel *原始*索引
        """
        B, S, D = lfm_output.shape
        device = lfm_output.device
        
        # --- 1. 一次性分离整个批次的特征 ---
        panel_mask = (element_types == self.layout_types['TYPE_PANEL'])    # (B, S)
        char_mask = (element_types == self.layout_types['TYPE_CHAR'])     # (B, S)
        dialog_mask = (element_types == self.layout_types['TYPE_DIALOG'])   # (B, S)
        
        panel_features = lfm_output[panel_mask]      # (TotalPanels, D)
        dialog_features = lfm_output[dialog_mask]    # (TotalDialogs, D)
        character_features = lfm_output[char_mask]   # (TotalChars, D)

        outputs = {}
    
        # --- 2. Panel Predictions (最简单) ---
        if panel_features.numel() > 0:
            outputs['panel_class_logits'] = self.panel_class_head(panel_features)
            outputs['panel_bbox'] = self.panel_bbox_head(panel_features)
            outputs['panel_offsets'] = self.panel_offsets_head(panel_features)
        
        # --- 3. Dialog & Character Predictions (核心步骤) ---
        if dialog_features.numel() > 0 or character_features.numel() > 0:
            # a. 找到所有 Panel 特征在批次内的位置
            # batch_idx_panels: (TotalPanels,)，值为 [0, 0, ..., 1, 1, ..., B-1]
            batch_idx_panels = torch.nonzero(panel_mask, as_tuple=True)[0]

            # b. 为每个子元素（dialog/char）找到其父 panel 的特征
            # 这是整个函数最关键的部分
            dialog_parent_feats = self._get_parent_features(
                dialog_mask, parent_panel_indices, panel_mask, element_indices, batch_idx_panels, panel_features
            )
            char_parent_feats = self._get_parent_features(
                char_mask, parent_panel_indices, panel_mask, element_indices, batch_idx_panels, panel_features
            )

            # c. Dialog Predictions
            if dialog_features.numel() > 0:
                outputs['dialog_bbox'] = self.element_bbox_head(dialog_features)
                bl, br = self.breakout_head(dialog_features, dialog_parent_feats)
                outputs['dialog_breakout_logits'] = bl
                outputs['dialog_breakout_ratio'] = br
                outputs['dialog_shape_logits'] = self.dialog_shape_head(dialog_features)

            # d. Character Predictions
            if character_features.numel() > 0:
                outputs['character_bbox'] = self.element_bbox_head(character_features)
                bl, br = self.breakout_head(character_features, char_parent_feats)
                outputs['character_breakout_logits'] = bl
                outputs['character_breakout_ratio'] = br

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
    
    # def forward(self, lfm_output, element_types, parent_panel_indices):
    #     """
    #         lfm_output: (B, S, D) - LFM 输出特征序列
    #         element_types: (B, S) - 每个 token 的类型
    #         element_indices: (B, S) - 每个 token 的自身索引 (0, 1, 2...)
    #         parent_panel_indices: (B, S) - 每个元素的父 Panel 原始索引
    #     """
    #     seq_len, d_model = lfm_output.shape
    #     device = lfm_output.device
        
    #     panel_mask = (element_types == 2)
    #     char_mask = (element_types == 3)
    #     dialog_mask = (element_types == 4)
        
    #     panel_features = lfm_output[panel_mask]    # (num_panels, d)
    #     dialog_features = lfm_output[dialog_mask]  # (num_dialogs, d)
    #     character_features = lfm_output[char_mask]      # (num_chars, d)

    #     # 初始化输出
    #     panel_outputs = {'class_logits': None, 'bbox': None, 'offsets': None}
    #     dialog_outputs = {'bbox': None, 'breakout_logits': None, 'breakout_ratio': None, 'shape_logits': None}
    #     character_outputs = {'bbox': None, 'breakout_logits': None, 'breakout_ratio': None}
    
    #     # Panel Predictions
    #     if panel_features.shape[0] > 0:
    #         panel_outputs['class_logits'] = self.panel_class_head(panel_features)  # (num_panels, num_classes)
    #         panel_outputs['bbox'] = self.panel_bbox_head(panel_features)  # (num_panels, 4)
    #         panel_outputs['offsets'] = self.panel_offsets_head(panel_features)  # (num_panels, 8)
        
    #     if dialog_features.shape[0] > 0:
    #         dialog_outputs['bbox'] = self.element_bbox_head(dialog_features)  # (num_dialogs, 4)

    #         # 获取父 Panel 特征（parent_panel_indices 已在 planner 里重映射）
    #         dialog_parent_indices = parent_panel_indices[dialog_mask]         # (num_dialogs,)
    #         if panel_features.shape[0] > 0 and dialog_parent_indices.numel() > 0:
    #             # 只对有效 parent >=0 的位置做融合；无效位置用 0 特征（不报错）
    #             valid_mask = (dialog_parent_indices >= 0)
    #             fused_parent = torch.zeros(dialog_features.shape[0], panel_features.shape[1], device=device)
    #             if valid_mask.any():
    #                 fused_parent[valid_mask] = panel_features[dialog_parent_indices[valid_mask]]
    #         else:
    #             fused_parent = torch.zeros_like(dialog_features)

    #         bl, br = self.breakout_head(dialog_features, fused_parent)
    #         dialog_outputs['breakout_logits'] = bl  # (num_dialogs, 1)
    #         dialog_outputs['breakout_ratio'] = br   # (num_dialogs, 1)
    #         dialog_outputs['shape_logits'] = self.dialog_shape_head(dialog_features)  # (num_dialogs, num_shapes)
        
    #     # Character Predictions
    #     if character_features.shape[0] > 0:
    #         character_outputs['bbox'] = self.element_bbox_head(character_features)  # (num_characters, 4)

    #         char_parent_indices = parent_panel_indices[char_mask]
    #         if panel_features.shape[0] > 0 and char_parent_indices.numel() > 0:
    #             valid_mask = (char_parent_indices >= 0)
    #             fused_parent = torch.zeros(character_features.shape[0], panel_features.shape[1], device=device)
    #             if valid_mask.any():
    #                 fused_parent[valid_mask] = panel_features[char_parent_indices[valid_mask]]
    #         else:
    #             fused_parent = torch.zeros_like(character_features)

    #         bl, br = self.breakout_head(character_features, fused_parent)
    #         character_outputs['breakout_logits'] = bl
    #         character_outputs['breakout_ratio'] = br
            
            
    #     return panel_outputs, dialog_outputs, character_outputs

