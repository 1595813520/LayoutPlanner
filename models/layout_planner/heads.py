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
        element_types: (seq_len,) - 每个 token 的类型 (0: PAD, 1: PAGE_CTRL, 2: PANEL, 3: CHARACTER, 4: DIALOG)
        parent_panel_indices: (seq_len,) - 每个 Dialog/Character 的父 Panel 索引
        """
        
        
        seq_len, d_model = lfm_output.shape
        device = lfm_output.device
        
        panel_mask = (element_types == 2)
        char_mask = (element_types == 3)
        dialog_mask = (element_types == 4)
        
        panel_features = lfm_output[panel_mask]    # (num_panels, d)
        dialog_features = lfm_output[dialog_mask]  # (num_dialogs, d)
        character_features = lfm_output[char_mask]      # (num_chars, d)

        # 初始化输出
        panel_outputs = {'class_logits': None, 'bbox': None, 'offsets': None}
        dialog_outputs = {'bbox': None, 'breakout_logits': None, 'breakout_ratio': None, 'shape_logits': None}
        character_outputs = {'bbox': None, 'breakout_logits': None, 'breakout_ratio': None}
    
        # Panel Predictions
        if panel_features.shape[0] > 0:
            panel_outputs['class_logits'] = self.panel_class_head(panel_features)  # (num_panels, num_classes)
            panel_outputs['bbox'] = self.panel_bbox_head(panel_features)  # (num_panels, 4)
            panel_outputs['offsets'] = self.panel_offsets_head(panel_features)  # (num_panels, 8)
        
        if dialog_features.shape[0] > 0:
            dialog_outputs['bbox'] = self.element_bbox_head(dialog_features)  # (num_dialogs, 4)

            # 获取父 Panel 特征（parent_panel_indices 已在 planner 里重映射）
            dialog_parent_indices = parent_panel_indices[dialog_mask]         # (num_dialogs,)
            if panel_features.shape[0] > 0 and dialog_parent_indices.numel() > 0:
                # 只对有效 parent >=0 的位置做融合；无效位置用 0 特征（不报错）
                valid_mask = (dialog_parent_indices >= 0)
                fused_parent = torch.zeros(dialog_features.shape[0], panel_features.shape[1], device=device)
                if valid_mask.any():
                    fused_parent[valid_mask] = panel_features[dialog_parent_indices[valid_mask]]
            else:
                fused_parent = torch.zeros_like(dialog_features)

            bl, br = self.breakout_head(dialog_features, fused_parent)
            dialog_outputs['breakout_logits'] = bl  # (num_dialogs, 1)
            dialog_outputs['breakout_ratio'] = br   # (num_dialogs, 1)
            dialog_outputs['shape_logits'] = self.dialog_shape_head(dialog_features)  # (num_dialogs, num_shapes)
        
        # Character Predictions
        if character_features.shape[0] > 0:
            character_outputs['bbox'] = self.element_bbox_head(character_features)  # (num_characters, 4)

            char_parent_indices = parent_panel_indices[char_mask]
            if panel_features.shape[0] > 0 and char_parent_indices.numel() > 0:
                valid_mask = (char_parent_indices >= 0)
                fused_parent = torch.zeros(character_features.shape[0], panel_features.shape[1], device=device)
                if valid_mask.any():
                    fused_parent[valid_mask] = panel_features[char_parent_indices[valid_mask]]
            else:
                fused_parent = torch.zeros_like(character_features)

            bl, br = self.breakout_head(character_features, fused_parent)
            character_outputs['breakout_logits'] = bl
            character_outputs['breakout_ratio'] = br
            
            
        return panel_outputs, dialog_outputs, character_outputs

    
class PredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.l1 = nn.L1Loss(reduction='none')

    def _reduced_masked_l1(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred: (K, D), target: (M, D) where K <= M (we assume predictions correspond to first K targets)
        We take M_first = target[:K] and compute mean L1 over elements.
        """
        if pred is None or pred.numel() == 0:
            return torch.tensor(0.0, device=target.device)
        K = pred.shape[0]
        tgt = target[:K]
        loss = self.l1(pred, tgt).mean()
        return loss

    def _masked_elementwise_bce(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: (K,) or (K,1)
        targets: (M,)  -> use first K elements
        """
        if logits is None or logits.numel() == 0:
            return torch.tensor(0.0, device=targets.device)
        K = logits.shape[0]
        tgt = targets[:K].float()
        logits_flat = logits.view(-1)
        loss = self.bce(logits_flat, tgt)
        return loss.mean()

    def forward(self, predictions, targets):
        """
        predictions: (panel_outputs, dialog_outputs, character_outputs) per-sample (not batched)
        targets: dict containing fixed-size tensors for this sample (from collate)
                 keys: panel_bboxes (Pmax,4), panel_offsets (Pmax,8), panel_classes (Pmax)
                       dialog_bboxes (Dmax,4), dialog_breakout_labels (Dmax), dialog_breakout_ratios (Dmax), dialog_shapes (Dmax)
                       character_bboxes (Cmax,4), character_breakout_labels (Cmax), character_breakout_ratios (Cmax)
                       panel_mask/dialog_mask/char_mask (Pmax/Dmax/Cmax) (0/1)
        """
        panel_out, dialog_out, char_out = predictions
        total = torch.tensor(0.0, device=targets['panel_bboxes'].device)
        loss_dict = {}

        # Panels: predictions have size K_panel (num_panels)
        if panel_out.get('bbox') is not None:
            # class CE (slice to K)
            cls_logits = panel_out['class_logits']               # (Kp, C)
            Kp = cls_logits.shape[0]
            panel_classes_target = targets['panel_classes'][:Kp]
            panel_class_loss = self.ce(cls_logits, panel_classes_target)
            total = total + panel_class_loss
            loss_dict['panel_class'] = float(panel_class_loss.item())

            # bbox L1
            bbox_loss = self._reduced_masked_l1(panel_out['bbox'], targets['panel_bboxes'])
            total = total + bbox_loss
            loss_dict['panel_bbox'] = float(bbox_loss.item())

            # offsets L1
            offsets_loss = self._reduced_masked_l1(panel_out['offsets'], targets['panel_offsets'])
            total = total + offsets_loss
            loss_dict['panel_offsets'] = float(offsets_loss.item())

        # Dialogs
        if dialog_out.get('bbox') is not None:
            dbbox_loss = self._reduced_masked_l1(dialog_out['bbox'], targets['dialog_bboxes'])
            total = total + dbbox_loss
            loss_dict['dialog_bbox'] = float(dbbox_loss.item())

            # breakout class (BCE)
            dbreak_loss = self._masked_elementwise_bce(dialog_out['breakout_logits'].squeeze(-1), targets['dialog_breakout_labels'])
            total = total + dbreak_loss
            loss_dict['dialog_breakout_class'] = float(dbreak_loss.item())

            # breakout ratio L1
            dratio_loss = self._reduced_masked_l1(dialog_out['breakout_ratio'].squeeze(-1), targets['dialog_breakout_ratios'])
            total = total + dratio_loss
            loss_dict['dialog_breakout_ratio'] = float(dratio_loss.item())

            # dialog shape CE
            dshape_logits = dialog_out.get('shape_logits')
            if dshape_logits is not None:
                Kd = dshape_logits.shape[0]
                dshape_target = targets['dialog_shapes'][:Kd]
                dshape_loss = self.ce(dshape_logits, dshape_target)
                total = total + dshape_loss
                loss_dict['dialog_shape'] = float(dshape_loss.item())

        # Characters
        if char_out.get('bbox') is not None:
            cbbox_loss = self._reduced_masked_l1(char_out['bbox'], targets['character_bboxes'])
            total = total + cbbox_loss
            loss_dict['character_bbox'] = float(cbbox_loss.item())

            cbreak_loss = self._masked_elementwise_bce(char_out['breakout_logits'].squeeze(-1), targets['character_breakout_labels'])
            total = total + cbreak_loss
            loss_dict['character_breakout_class'] = float(cbreak_loss.item())

            cratio_loss = self._reduced_masked_l1(char_out['breakout_ratio'].squeeze(-1), targets['character_breakout_ratios'])
            total = total + cratio_loss
            loss_dict['character_breakout_ratio'] = float(cratio_loss.item())

        return total, loss_dict