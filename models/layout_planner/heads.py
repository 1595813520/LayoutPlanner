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
    
    def forward(self, x, parent_panel_features):
        # 融合当前元素和父 Panel 特征
        fused_features = torch.cat([x, parent_panel_features], dim=-1)
        breakout_logit = self.mlp_class(fused_features)  # Shape: (num_elements, 1)
        breakout_ratio = self.mlp_ratio(fused_features)  # Shape: (num_elements, 1)
        return breakout_logit, breakout_ratio

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
    def __init__(self, d_model=512, num_panel_classes=4, num_dialog_shapes=5):
        super().__init__()
        # Panel Prediction Heads
        self.panel_class_head = PanelClassHead(d_model, num_panel_classes)
        self.panel_bbox_head = PanelBBoxHead(d_model)
        self.panel_offsets_head = PanelOffsetsHead(d_model)
        
        # Dialog/Character Prediction Heads
        self.element_bbox_head = ElementBBoxHead(d_model)
        self.breakout_head = BreakoutHead(d_model)
        self.dialog_shape_head = DialogShapeHead(d_model, num_dialog_shapes)
    
    def forward(self, lfm_output, element_types, parent_panel_indices):
        """
        lfm_output: (seq_len, d_model) - LFM 输出特征序列
        element_types: (seq_len,) - 每个 token 的类型 (0: PAGE_CTRL, 1: PANEL, 2: DIALOG, 3: CHARACTER, 4: PAD)
        parent_panel_indices: (seq_len,) - 每个 Dialog/Character 的父 Panel 索引
        """
        seq_len, d_model = lfm_output.shape
        device = lfm_output.device
        
        # 初始化输出
        panel_outputs = {'class_logits': None, 'bbox': None, 'offsets': None}
        dialog_outputs = {'bbox': None, 'breakout_logits': None, 'breakout_ratio': None, 'shape_logits': None}
        character_outputs = {'bbox': None, 'breakout_logits': None, 'breakout_ratio': None}
        
        # 分离特征向量
        panel_mask = (element_types == 1)  # PANEL tokens
        dialog_mask = (element_types == 2)  # DIALOG tokens
        character_mask = (element_types == 3)  # CHARACTER tokens
        
        panel_features = lfm_output[panel_mask]  # Shape: (num_panels, d_model)
        dialog_features = lfm_output[dialog_mask]  # Shape: (num_dialogs, d_model)
        character_features = lfm_output[character_mask]  # Shape: (num_characters, d_model)
        
        # Panel Predictions
        if panel_features.shape[0] > 0:
            panel_outputs['class_logits'] = self.panel_class_head(panel_features)  # (num_panels, num_classes)
            panel_outputs['bbox'] = self.panel_bbox_head(panel_features)  # (num_panels, 4)
            panel_outputs['offsets'] = self.panel_offsets_head(panel_features)  # (num_panels, 8)
        
        # Dialog Predictions
        if dialog_features.shape[0] > 0:
            dialog_outputs['bbox'] = self.element_bbox_head(dialog_features)  # (num_dialogs, 4)
            # 获取父 Panel 特征
            dialog_parent_indices = parent_panel_indices[dialog_mask]
            dialog_parent_features = panel_features[dialog_parent_indices]  # (num_dialogs, d_model)
            dialog_outputs['breakout_logits'], dialog_outputs['breakout_ratio'] = self.breakout_head(
                dialog_features, dialog_parent_features
            )  # (num_dialogs, 1), (num_dialogs, 1)
            dialog_outputs['shape_logits'] = self.dialog_shape_head(dialog_features)  # (num_dialogs, num_shapes)
        
        # Character Predictions
        if character_features.shape[0] > 0:
            character_outputs['bbox'] = self.element_bbox_head(character_features)  # (num_characters, 4)
            # 获取父 Panel 特征
            character_parent_indices = parent_panel_indices[character_mask]
            character_parent_features = panel_features[character_parent_indices]  # (num_characters, d_model)
            character_outputs['breakout_logits'], character_outputs['breakout_ratio'] = self.breakout_head(
                character_features, character_parent_features
            )  # (num_characters, 1), (num_characters, 1)
        
        return panel_outputs, dialog_outputs, character_outputs

class PredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        predictions: (panel_outputs, dialog_outputs, character_outputs)
        targets: Dictionary containing ground truth labels
        """
        panel_outputs, dialog_outputs, character_outputs = predictions
        total_loss = 0.0
        loss_dict = {}
        
        # Panel Losses
        if panel_outputs['class_logits'] is not None:
            panel_class_loss = self.ce_loss(
                panel_outputs['class_logits'], targets['panel_classes']
            )
            panel_bbox_loss = self.l1_loss(
                panel_outputs['bbox'], targets['panel_bboxes']
            )
            panel_offsets_loss = self.l1_loss(
                panel_outputs['offsets'], targets['panel_offsets']
            )
            total_loss += panel_class_loss + panel_bbox_loss + panel_offsets_loss
            loss_dict['panel_class'] = panel_class_loss.item()
            loss_dict['panel_bbox'] = panel_bbox_loss.item()
            loss_dict['panel_offsets'] = panel_offsets_loss.item()
        
        # Dialog Losses
        if dialog_outputs['bbox'] is not None:
            dialog_bbox_loss = self.l1_loss(
                dialog_outputs['bbox'], targets['dialog_bboxes']
            )
            dialog_breakout_class_loss = self.bce_loss(
                dialog_outputs['breakout_logits'].squeeze(-1), targets['dialog_breakout_labels'].float()
            )
            dialog_breakout_ratio_loss = self.l1_loss(
                dialog_outputs['breakout_ratio'].squeeze(-1), targets['dialog_breakout_ratios']
            )
            dialog_shape_loss = self.ce_loss(
                dialog_outputs['shape_logits'], targets['dialog_shapes']
            )
            total_loss += dialog_bbox_loss + dialog_breakout_class_loss + dialog_breakout_ratio_loss + dialog_shape_loss
            loss_dict['dialog_bbox'] = dialog_bbox_loss.item()
            loss_dict['dialog_breakout_class'] = dialog_breakout_class_loss.item()
            loss_dict['dialog_breakout_ratio'] = dialog_breakout_ratio_loss.item()
            loss_dict['dialog_shape'] = dialog_shape_loss.item()
        
        # Character Losses
        if character_outputs['bbox'] is not None:
            character_bbox_loss = self.l1_loss(
                character_outputs['bbox'], targets['character_bboxes']
            )
            character_breakout_class_loss = self.bce_loss(
                character_outputs['breakout_logits'].squeeze(-1), targets['character_breakout_labels'].float()
            )
            character_breakout_ratio_loss = self.l1_loss(
                character_outputs['breakout_ratio'].squeeze(-1), targets['character_breakout_ratios']
            )
            total_loss += character_bbox_loss + character_breakout_class_loss + character_breakout_ratio_loss
            loss_dict['character_bbox'] = character_bbox_loss.item()
            loss_dict['character_breakout_class'] = character_breakout_class_loss.item()
            loss_dict['character_breakout_ratio'] = character_breakout_ratio_loss.item()
        
        return total_loss, loss_dict