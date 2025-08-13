
# ==============================================================================
# 文件: models/training/style_calculator.py
# 作用: (在线模块) 作为模型的一部分，在训练时计算预测布局的风格向量 S_pred。
#       完全使用PyTorch实现，保证端到端可微分。
# ==============================================================================
import torch
import torch.nn as nn

class StyleCalculator(nn.Module):
    """
    用PyTorch实现风格参数计算，使其完全可微分。
    注意：此模块的输入是模型直接预测的、批处理过的张量。
    """
    def __init__(self, page_width, page_height, delta_ratio=0.01):
        super().__init__()
        self.page_width = page_width
        self.page_height = page_height
        self.delta = page_width * delta_ratio

        # 形状类型附加分
        # 将bonus_map转换为一个tensor，以便在计算图中使用
        # 假设shape_type被编码为 0:rect, 1:trapezoid, ...
        self.bonus_tensor = torch.tensor([0.0, 0.2, 0.3, 0.4, 0.6, 0.6])

    def _shoelace_area_torch(self, coords):
        # coords shape: (B, N, V, 2), V是顶点数
        x = coords[..., 0]
        y = coords[..., 1]
        # (x_i * y_{i+1} - x_{i+1} * y_i)
        area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1, dims=-1) - torch.roll(x, -1, dims=-1) * y, dim=-1))
        return area

    def forward(self, pred_layout):
        """
        Args:
            pred_layout (dict): 包含模型预测结果的字典，所有值都是Tensor。
                - 'panel_four_points': (B, MaxPanels, 4, 2)
                - 'panel_bbox': (B, MaxPanels, 4)
                - 'panel_shape_type_idx': (B, MaxPanels)
                - 'char_bbox': (B, MaxChars, 4)
                - 'char_breakout_ratio': (B, MaxChars)
                - 'dialog_bbox': (B, MaxDialogs, 4)
                - 'dialog_breakout_ratio': (B, MaxDialogs)
                - 'panel_mask': (B, MaxPanels) # 布尔掩码，标记哪些是真实panel
        
        Returns:
            torch.Tensor: (B, 4) 的风格向量 S_pred
        """
        # --- 1. 布局密度 (LD) ---
        # 警告: 可微分凸包是主要难点。在实际项目中，需要使用如
        # `torch-points3d` 或 `torch-cluster` 等库，或者实现可微的近似算法。
        # 此处为了演示逻辑，我们使用一个占位符。
        # 作为一个可行的、完全可微的近似，可以用所有panel的最小外接矩形面积代替凸包面积。
        panel_areas = self._shoelace_area_torch(pred_layout['panel_four_points'])
        total_panel_area = torch.sum(panel_areas * pred_layout['panel_mask'], dim=1)
        # Placeholder for hull area
        hull_area = torch.full_like(total_panel_area, self.page_width * self.page_height * 0.8) # 假设凸包占页面80%
        layout_density = total_panel_area / (hull_area + 1e-6)

        # --- 2. 对齐度 (AS) ---
        bbox = pred_layout['panel_bbox']
        x_min, y_min, x_max, y_max = bbox.unbind(dim=-1)
        key_coords = torch.stack([
            x_min, (x_min + x_max) / 2, x_max,
            y_min, (y_min + y_max) / 2, y_max
        ], dim=-1) # (B, MaxPanels, 6)
        
        # 使用广播和torch.cdist计算成对距离会更高效，这里用循环以保持逻辑清晰
        dist_matrix = torch.abs(key_coords.unsqueeze(2) - key_coords.unsqueeze(1)) # (B, P, P, 6)
        
        # 用一个很大的值填充对角线，以忽略自身
        dist_matrix.diagonal(dim1=1, dim2=2).fill_(float('inf'))
        
        min_dists, _ = torch.min(dist_matrix, dim=2)
        align_scores = torch.exp(-min_dists / self.delta)
        
        # 只对真实panel计算平均分
        masked_scores = align_scores * pred_layout['panel_mask'].unsqueeze(-1)
        num_real_panels = pred_layout['panel_mask'].sum(dim=1).unsqueeze(-1) * 6
        alignment_score = torch.sum(masked_scores, dim=[1, 2]) / (num_real_panels.squeeze() + 1e-6)

        # --- 3. 形态不稳定性 (SI) ---
        coords = pred_layout['panel_four_points']
        # 计算周长
        rolled_coords = torch.roll(coords, -1, dims=2)
        perimeters = torch.sum(torch.norm(coords - rolled_coords, dim=-1), dim=2)
        
        areas_si = self._shoelace_area_torch(coords)
        circularity = (4 * torch.pi * areas_si) / (perimeters.pow(2) + 1e-6)
        geom_irregularity = 1 - circularity
        
        # 获取类型附加分
        bonus = self.bonus_tensor.to(geom_irregularity.device)[pred_layout['panel_shape_type_idx']]
        instability = geom_irregularity + bonus
        
        # 面积加权平均
        weighted_instability = torch.sum((areas_si * instability) * pred_layout['panel_mask'], dim=1)
        total_area_for_si = torch.sum(areas_si * pred_layout['panel_mask'], dim=1)
        shape_instability = weighted_instability / (total_area_for_si + 1e-6)

        # --- 4. 破格强度 (BI) ---
        char_bbox = pred_layout['char_bbox']
        char_areas = (char_bbox[..., 2] - char_bbox[..., 0]) * (char_bbox[..., 3] - char_bbox[..., 1])
        char_breakout_area = torch.sum(char_areas * pred_layout['char_breakout_ratio'], dim=1)
        
        dialog_bbox = pred_layout['dialog_bbox']
        dialog_areas = (dialog_bbox[..., 2] - dialog_bbox[..., 0]) * (dialog_bbox[..., 3] - dialog_bbox[..., 1])
        dialog_breakout_area = torch.sum(dialog_areas * pred_layout['dialog_breakout_ratio'], dim=1)
        
        total_breakout_area = char_breakout_area + dialog_breakout_area
        breakout_intensity = total_breakout_area / (self.page_width * self.page_height)

        # 将4个标量堆叠成一个向量
        return torch.stack([layout_density, alignment_score, shape_instability, breakout_intensity], dim=1)

