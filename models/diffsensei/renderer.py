import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple

class LayoutToMaskRenderer:
    def __init__(self, canvas_size: Tuple[int, int] = (1448, 1024), latent_size: Tuple[int, int] = (181, 128)):
        """
        初始化 Renderer
        canvas_size: 页面画布尺寸 (height, width)，例如 (1448, 1024)
        latent_size: U-Net 潜空间尺寸 (height, width)，例如 (181, 128)
        """
        self.canvas_size = canvas_size  # (H, W)
        self.latent_size = latent_size  # (H_latent, W_latent)
        self.scale_factor = (latent_size[0] / canvas_size[0], latent_size[1] / canvas_size[1])

    def create_panel_mask(self, four_points: List[List[int]]) -> np.ndarray:
        """
        根据 four_points 创建 Panel 形状掩码
        four_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        mask = np.zeros(self.canvas_size, dtype=np.uint8)
        points = np.array(four_points, dtype=np.int32).reshape(-1, 2)
        # 确保形成闭合多边形
        cv2.fillPoly(mask, [points], color=1)
        return mask

    def create_dialog_mask(self, bbox: List[float], shape_type: str, panel_mask: np.ndarray) -> np.ndarray:
        """
        根据 Dialog 的 BBox 和 shape_type 创建掩码
        bbox: [x_center, y_center, width, height] (归一化 [0, 1])
        shape_type: bubble_oval, bubble_burst, scene_text
        panel_mask: 父 Panel 的掩码，用于裁剪
        """
        H, W = self.canvas_size
        # 将归一化 BBox 转换为像素坐标
        x_center, y_center, width, height = bbox
        x_center, y_center = int(x_center * W), int(y_center * H)
        width, height = int(width * W), int(height * H)
        x1, y1 = x_center - width // 2, y_center - height // 2
        x2, y2 = x_center + width // 2, y_center + height // 2

        mask = np.zeros(self.canvas_size, dtype=np.uint8)

        if shape_type == 'bubble_oval':
            # 绘制椭圆
            center = (x_center, y_center)
            axes = (width // 2, height // 2)
            cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)
        
        elif shape_type == 'bubble_burst':
            # 模拟爆炸形：使用星形模板并缩放
            points = self._generate_star_shape(x_center, y_center, width, height)
            cv2.fillPoly(mask, [points], color=1)
        
        elif shape_type == 'scene_text':
            # 仅保留 BBox 区域作为文本区域
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=1, thickness=-1)
        
        else:
            # 默认：矩形
            cv2.rectangle(mask, (x1, y1), (x2, y2), color=1, thickness=-1)

        # 裁剪到父 Panel 内部
        mask = mask & panel_mask
        return mask

    def _generate_star_shape(self, x_center: int, y_center: int, width: int, height: int) -> np.ndarray:
        """生成爆炸形（星形）模板，缩放到 BBox"""
        # 简单星形：10 个点（5 个外点，5 个内点）
        num_points = 5
        outer_radius = min(width, height) // 2
        inner_radius = outer_radius // 2
        angles = np.linspace(0, 2 * np.pi, num_points * 2, endpoint=False)
        points = []
        for i, angle in enumerate(angles):
            r = outer_radius if i % 2 == 0 else inner_radius
            x = int(x_center + r * np.cos(angle))
            y = int(y_center + r * np.sin(angle))
            points.append([x, y])
        return np.array(points, dtype=np.int32)

    def create_character_mask(self, bbox: List[float], panel_mask: np.ndarray) -> np.ndarray:
        """
        根据 Character 的 BBox 创建掩码
        bbox: [x_center, y_center, width, height] (归一化 [0, 1])
        panel_mask: 父 Panel 的掩码，用于裁剪
        """
        H, W = self.canvas_size
        x_center, y_center, width, height = bbox
        x_center, y_center = int(x_center * W), int(y_center * H)
        width, height = int(width * W), int(height * H)
        x1, y1 = x_center - width // 2, y_center - height // 2
        x2, y2 = x_center + width // 2, y_center + height // 2

        mask = np.zeros(self.canvas_size, dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=1, thickness=-1)
        mask = mask & panel_mask
        return mask

    def resize_mask_to_latent(self, mask: np.ndarray) -> np.ndarray:
        """将掩码降采样到潜空间分辨率"""
        return cv2.resize(mask, (self.latent_size[1], self.latent_size[0]), interpolation=cv2.INTER_NEAREST)

    def render_layout(self, layout: Dict) -> Dict:
        """
        渲染整个布局，生成所有掩码
        layout: Layout Planner 输出，包含 panels, dialogs, characters
        """
        H, W = self.canvas_size
        output = {
            'panel_masks': [],  # List of (H, W) binary masks for each panel
            'dialog_masks': [],  # List of (H, W) binary masks for each panel's dialogs
            'character_masks': [],  # List of (H, W) binary masks for each panel's characters
            'latent_panel_masks': []  # List of (H_latent, W_latent) masks for U-Net
        }

        for i, panel in enumerate(layout['panels']):
            # 1. 创建 Panel 形状掩码
            panel_mask = self.create_panel_mask(panel['four_points'])
            output['panel_masks'].append(panel_mask)
            
            # 2. 创建 Dialog 合并掩码
            dialog_mask = np.zeros(self.canvas_size, dtype=np.uint8)
            for dialog in layout['dialogs']:
                if dialog['parent_panel_idx'] == i:
                    dialog_submask = self.create_dialog_mask(
                        dialog['bbox'], dialog['shape_type'], panel_mask
                    )
                    dialog_mask |= dialog_submask
            output['dialog_masks'].append(dialog_mask)
            
            # 3. 创建 Character 合并掩码
            character_mask = np.zeros(self.canvas_size, dtype=np.uint8)
            for character in layout['characters']:
                if character['parent_panel_idx'] == i:
                    character_submask = self.create_character_mask(character['bbox'], panel_mask)
                    character_mask |= character_submask
            output['character_masks'].append(character_mask)
            
            # 4. 降采样 Panel 掩码到潜空间分辨率
            latent_panel_mask = self.resize_mask_to_latent(panel_mask)
            output['latent_panel_masks'].append(latent_panel_mask)

        return output

    def zero_shot_guidance(self, xt: torch.Tensor, masks: List[np.ndarray], scheduler, timestep: int, xbg: torch.Tensor = None) -> torch.Tensor:
        """
        Zero-shot 布局约束：将生成内容限制在 Panel 掩码内
        xt: 当前时间步的潜空间张量 (batch, channels, H_latent, W_latent)
        masks: List of Panel 掩码 (H_latent, W_latent)
        scheduler: 扩散调度器
        timestep: 当前时间步
        xbg: 背景潜空间 (batch, channels, H_latent, W_latent)，若为 None 则随机初始化
        """
        batch_size, channels, H_latent, W_latent = xt.shape
        if xbg is None:
            xbg = torch.randn_like(xt)
        
        # 合并所有 Panel 掩码
        combined_mask = np.zeros(self.latent_size, dtype=np.uint8)
        for mask in masks:
            combined_mask |= mask
        
        # 转换为张量
        mask_tensor = torch.tensor(combined_mask, dtype=torch.float32, device=xt.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H_latent, W_latent)
        
        # 引导式混合
        x_blended = xt * mask_tensor + xbg * (1 - mask_tensor)
        
        # 重新加噪
        noise = torch.randn_like(x_blended)
        x_blended = x_blended + scheduler.add_noise(x_blended, noise, timestep)
        
        return x_blended

# 示例用法
if __name__ == "__main__":
    # 模拟 Layout Planner 输出
    layout = {
        'panels': [
            {
                'four_points': [[50, 100], [410, 100], [450, 700], [50, 700]],
                'shape_type': 'panel_trapezoid',
                'offsets': [0, 0, -40, 0, 0, 0, 0, 0]
            },
            {
                'four_points': [[500, 100], [950, 100], [950, 450], [500, 450]],
                'shape_type': 'panel_rect',
                'offsets': [0, 0, 0, 0, 0, 0, 0, 0]
            }
        ],
        'dialogs': [
            {
                'bbox': [0.3, 0.2, 0.2, 0.1],
                'shape_type': 'bubble_oval',
                'parent_panel_idx': 0
            },
            {
                'bbox': [0.6, 0.3, 0.15, 0.08],
                'shape_type': 'bubble_burst',
                'parent_panel_idx': 0
            }
        ],
        'characters': [
            {
                'bbox': [0.4, 0.5, 0.3, 0.4],
                'parent_panel_idx': 0
            }
        ]
    }

    # 初始化 Renderer
    renderer = LayoutToMaskRenderer(canvas_size=(1448, 1024), latent_size=(181, 128))
    
    # 渲染掩码
    masks = renderer.render_layout(layout)
    
    # 模拟 U-Net 潜空间
    xt = torch.randn(1, 4, 181, 128)  # (batch, channels, H_latent, W_latent)
    class DummyScheduler:
        def add_noise(self, x, noise, timestep):
            return x + noise * 0.1  # 简化模拟
    
    scheduler = DummyScheduler()
    x_blended = renderer.zero_shot_guidance(
        xt, masks['latent_panel_masks'], scheduler, timestep=0
    )
    
    print("Generated masks:")
    print(f"Panel masks: {len(masks['panel_masks'])}")
    print(f"Dialog masks: {len(masks['dialog_masks'])}")
    print(f"Character masks: {len(masks['character_masks'])}")
    print(f"Latent panel masks: {len(masks['latent_panel_masks'])}")