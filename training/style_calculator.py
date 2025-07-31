import torch
import torch.nn as nn
# from torch_geometric.nn import convex_hull # 假设我们有一个可微分的凸包库

class DifferentiableStyleCalculator(nn.Module):
    """
    用PyTorch实现5个风格参数的计算，使其完全可微分。
    """
    def __init__(self):
        super().__init__()

    def forward(self, predicted_layout):
        """
        输入: 模型预测的布局字典
        输出: 5维的风格向量 S_pred
        """
        # 从预测布局中解析出panels和dialogs的坐标和类型
        # 此处为伪代码，具体解析方式取决于predicted_layout的结构
        panels_coords = predicted_layout['panels']['coords'] # shape: (B, N_panel, 4, 2)
        dialogs_coords = predicted_layout['dialogs']['coords'] # shape: (B, N_dialog, 4, 2)
        # ... 其他属性

        # 分别计算5个参数
        ld = self.calculate_ld(panels_coords)
        as_score = self.calculate_as(panels_coords)
        si = self.calculate_si(panels_coords, dialogs_coords, predicted_layout)
        bi = self.calculate_bi(predicted_layout)
        sp = self.calculate_sp(predicted_layout)
        
        # 将5个标量堆叠成一个向量
        # unsqueeze(1)是为了让shape变为 [Batch, 1]
        style_vector_pred = torch.stack([ld, as_score, si, bi, sp], dim=1) # shape: [Batch, 5]
        return style_vector_pred

    def calculate_polygon_area(self, coords):
        # 使用Shoelace公式计算多边形面积，此公式可微
        # coords shape: (B, N, V, 2), V是顶点数
        x = coords[..., 0]
        y = coords[..., 1]
        # (x_i * y_{i+1} - x_{i+1} * y_i)
        return 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1, dims=-1) - torch.roll(x, -1, dims=-1) * y, dim=-1))

    def calculate_ld(self, panels_coords):
        # --- 布局密度 (Layout Density - LD) ---
        # 警告: 凸包计算是主要难点。这里使用占位符。
        # 在实际项目中，需要找到或实现一个可微分的凸包算法。
        # 作为一个可行的近似，可以用所有panel的最小外接矩形来代替凸包。
        panel_areas = self.calculate_polygon_area(panels_coords)
        sum_panel_areas = torch.sum(panel_areas, dim=1)

        # Placeholder for convex hull area
        # A_conv_hull = differentiable_convex_hull(panels_coords)
        # 这里我们用一个简化的近似：所有Panel的并集的包围盒
        min_x = torch.min(panels_coords[..., 0])
        max_x = torch.max(panels_coords[..., 0])
        # ... similar for y
        A_conv_hull_approx = (max_x - min_x) * (torch.max(panels_coords[..., 1]) - torch.min(panels_coords[..., 1]))
        
        ld = sum_panel_areas / (A_conv_hull_approx + 1e-6)
        return ld.mean() # 返回批次的平均值

    def calculate_as(self, panels_coords, delta=0.01):
        # --- 对齐度 (Alignment Score - AS) ---
        # panels_coords: (B, N, 4, 2) -> 我们需要中心点和边界
        # (B, N, 6), 6个基准: left, h_center, right, top, v_center, bottom
        centers = torch.mean(panels_coords, dim=2)
        left, top = torch.min(panels_coords, dim=2).values
        right, bottom = torch.max(panels_coords, dim=2).values
        
        key_coords = torch.stack([
            left[..., 0], centers[..., 0], right[..., 0],
            top[..., 1], centers[..., 1], bottom[..., 1]
        ], dim=-1)

        # 计算所有Panel对之间的距离
        dist_matrix = torch.cdist(key_coords, key_coords) # (B, N, N, 6)
        
        # 忽略对角线（自己和自己）
        dist_matrix.diagonal(dim1=1, dim2=2).fill_(float('inf'))

        # 找到每个Panel每个基准的最近距离
        min_dists, _ = torch.min(dist_matrix, dim=2) # (B, N, 6)
        
        # 计算对齐分数
        alignment_scores = torch.exp(-min_dists / delta)
        
        # 在所有Panel和所有基准上求平均
        as_score = torch.mean(alignment_scores)
        return as_score

    # ... 其他参数 (si, bi, sp) 的可微分实现 ...
    # si的实现会涉及到周长和查表，但逻辑是相似的。