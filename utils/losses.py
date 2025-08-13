import torch
import torch.nn as nn
import torch.nn.functional as F
from .style_calculator import DifferentiableStyleCalculator

class LayoutCompositeLoss(nn.Module):
    """
    复合损失函数，包含几何重建损失和风格一致性损失。
    """
    def __init__(self, lambda_style=0.5):
        super().__init__()
        self.lambda_style = lambda_style
        self.style_calculator = DifferentiableStyleCalculator()
        print(f"Composite Loss initialized with lambda_style = {self.lambda_style}")

    def forward(self, predicted_layout, ground_truth_batch):
        # --- 1. 几何重建损失 (L_geom) ---
        # 伪代码：根据你的数据结构，计算所有几何属性的损失
        # a. Panel 坐标损失
        gt_panel_coords = ground_truth_batch['panels']['coords']
        pred_panel_coords = predicted_layout['panels']['coords']
        loss_panel_coords = F.l1_loss(pred_panel_coords, gt_panel_coords)

        # b. Dialog BBox 损失
        gt_dialog_bbox = ground_truth_batch['dialogs']['bbox']
        pred_dialog_bbox = predicted_layout['dialogs']['bbox']
        loss_dialog_bbox = F.l1_loss(pred_dialog_bbox, gt_dialog_bbox)

        # c. 形状类型损失
        gt_shape_type = ground_truth_batch['elements']['shape_type']
        pred_shape_logits = predicted_layout['elements']['shape_logits']
        loss_shape_type = F.cross_entropy(pred_shape_logits.transpose(1, 2), gt_shape_type)
        
        # 合并所有几何损失
        L_geom = loss_panel_coords + loss_dialog_bbox + loss_shape_type

        # --- 2. 风格一致性损失 (L_style) ---
        # a. 从真值batch中获取风格向量
        S_gt = ground_truth_batch['style_vector'] # shape: [Batch, 5]
        
        # b. 从预测布局中计算风格向量
        S_pred = self.style_calculator(predicted_layout) # shape: [Batch, 5]
        
        # c. 计算MSE损失
        L_style = F.mse_loss(S_pred, S_gt)

        # --- 3. 计算总损失 ---
        L_total = L_geom + self.lambda_style * L_style

        # 返回总损失和用于日志记录的各分项损失
        loss_dict = {
            "total_loss": L_total.item(),
            "geom_loss": L_geom.item(),
            "style_loss": L_style.item(),
            "panel_coord_loss": loss_panel_coords.item(),
        }
        return L_total, loss_dict