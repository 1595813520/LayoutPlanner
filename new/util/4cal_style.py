
import json
import os
import numpy as np
from shapely.geometry import Polygon, MultiPoint

def calculate_style_parameters_offline(annotation):
    """
    根据单个页面的标注，计算四个风格参数。

    Args:
        annotation (dict): 单个页面的标注数据。

    Returns:
        dict: 包含四个风格参数的字典。
    """
    page_width = annotation['width']
    page_height = annotation['height']
    frames = annotation['frames']

    if not frames:
        return {
            "layout_density": 0.0,
            "alignment_score": 0.0,
            "shape_instability": 0.0,
            "breakout_intensity": 0.0
        }

    # --- 1. 布局密度 (Layout Density - LD) ---
    all_panel_points = []
    total_panel_area = 0.0
    for frame in frames:
        # 使用 four_points 进行精确的几何计算
        poly = Polygon(frame['four_points'])
        total_panel_area += poly.area
        all_panel_points.extend(frame['four_points'])
    
    # 计算所有panel顶点集合的凸包面积
    if len(all_panel_points) >= 3:
        hull_poly = MultiPoint(all_panel_points).convex_hull
        hull_area = hull_poly.area
    else:
        hull_area = 0.0
    
    layout_density = total_panel_area / hull_area if hull_area > 1e-6 else 0.0

    # --- 2. 对齐度 (Alignment Score - AS) ---
    # 使用 bbox 进行对齐判断，更符合规整性的直觉
    key_coords = []
    for frame in frames:
        x_min, y_min, x_max, y_max = frame['bbox']
        key_coords.append([
            x_min, (x_min + x_max) / 2, x_max,  # left, h_center, right
            y_min, (y_min + y_max) / 2, y_max   # top, v_center, bottom
        ])
    
    key_coords = np.array(key_coords)
    num_panels = len(frames)
    total_align_score = 0.0
    delta = page_width * 0.01  # 敏感度阈值

    if num_panels > 1:
        for i in range(num_panels):
            for j in range(6):  # 遍历6个关键坐标
                # 计算第i个panel的第j个关键坐标与其他所有panel对应坐标的距离
                dists = np.abs(key_coords[:, j] - key_coords[i, j])
                # 忽略自身
                min_dist = np.min(dists[np.arange(num_panels) != i])
                total_align_score += np.exp(-min_dist / delta)
        
        alignment_score = total_align_score / (num_panels * 6)
    else:
        alignment_score = 0.0 # 单个panel无对齐可言

    # --- 3. 形态不稳定性 (Shape Instability - SI) ---
    total_weighted_instability = 0.0
    total_area_for_si = 0.0
    
    # 形状类型附加分
    bonus_map = {
        'panel_rect': 0.0,
        'panel_trapezoid': 0.2,
        'panel_triangle': 0.4,
        'panel_irregular_quad': 0.6
    }

    for frame in frames:
        poly = Polygon(frame['four_points'])
        area = poly.area
        perimeter = poly.length
        
        if perimeter > 1e-6:
            # 几何不规则度
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            geom_irregularity = 1 - circularity
        else:
            geom_irregularity = 0.0
            
        bonus = bonus_map.get(frame['shape_type'], 0.6)
        instability = geom_irregularity + bonus
        
        total_weighted_instability += area * instability
        total_area_for_si += area

    shape_instability = total_weighted_instability / total_area_for_si if total_area_for_si > 1e-6 else 0.0

    # --- 4. 破格强度 (Breakout Intensity - BI) ---
    total_breakout_area = 0.0
    for frame in frames:
        for char in frame.get('characters', []):
            x_min, y_min, x_max, y_max = char['bbox']
            char_area = (x_max - x_min) * (y_max - y_min)
            total_breakout_area += char_area * char.get('breakout_ratio', 0.0)
            
        for dialog in frame.get('dialogs', []):
            # 注意：这里使用dialog的rect_box，而非dialog_bbox
            rect_box = dialog['rect_box']
            # 修复错误：处理可能的嵌套列表结构，确保获取数值
            if isinstance(rect_box[0], list):
                rect_box = [coord for sublist in rect_box for coord in sublist]
            x_min, y_min, x_max, y_max = rect_box
            dialog_area = (x_max - x_min) * (y_max - y_min)
            total_breakout_area += dialog_area * dialog.get('breakout_ratio', 0.0)

    page_area = page_width * page_height
    breakout_intensity = total_breakout_area / page_area if page_area > 1e-6 else 0.0

    return {
        "layout_density": float(layout_density),
        "alignment_score": float(alignment_score),
        "shape_instability": float(shape_instability),
        "breakout_intensity": float(breakout_intensity)
    }

# --- 使用示例 ---
if __name__ == '__main__':
    ANN_FILE = "DiffSensei-main/checkpoints/mangazero/f_annotations_with_breakout_new.json"
    OUT_ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_style.json"
    
    with open(ANN_FILE, 'r') as f:
        all_annotations = json.load(f)
    
    for page_annotation in all_annotations:
        style_params = calculate_style_parameters_offline(page_annotation)
        page_annotation['style_parameters'] = style_params
        
    os.makedirs(os.path.dirname(OUT_ANN_FILE), exist_ok=True)
    with open(OUT_ANN_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, indent=2)
        
    print(f"处理完成，结果保存至: {OUT_ANN_FILE}")
