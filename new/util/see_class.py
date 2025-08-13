import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from tqdm import tqdm

# 形状名称映射（保持不变）
bubble_shape_map = {
    'bubble_rect': "Rectangle",
    'bubble_oval': "Oval",
    'bubble_burst': "Burst",
    'bubble_flower': "Flower",
    'bubble_irregular': "Irregular"
}

panel_map = {
    'panel_rect': "Rectangle",
    'panel_trapezoid': "Trapezoid",
    'panel_triangle': "Triangle",
    'panel_irregular_quad': "irregular_quad",
}

# ########### 颜色方案调整：panel统一为一种颜色 ###########
PANEL_COLOR = (100, 100, 255)  # 蓝色
BUBBLE_COLOR = (0, 255, 0)  # 绿色

IMG_ROOT = '/data/DiffSensei-main/checkpoints/mangazero/images'
ANN_FILE = '/data/DiffSensei-main/checkpoints/mangazero/f_annotations_4pts.json'
MASK_VIS_ROOT = '/data/DiffSensei-main/checkpoints/mangazero/image_mask_vis_refine'
OUT_ROOT = '/data/DiffSensei-main/checkpoints/mangazero/image_4pts'
os.makedirs(OUT_ROOT, exist_ok=True)

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

def get_font(size=36):
    """尝试加载更好的字体，如果失败则使用默认字体"""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            font = ImageFont.load_default()
    return font

def draw_en_text(img, text, xy, size=36, color=(255,255,0)):
    """绘制带描边的文字（保持不变）"""
    draw = ImageDraw.Draw(img)
    font = get_font(size)
    x, y = xy
    
    # 绘制黑色描边
    stroke_width = max(1, size // 18)
    for dx in [-stroke_width, 0, stroke_width]:
        for dy in [-stroke_width, 0, stroke_width]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    
    # 绘制主文字
    draw.text(xy, text, font=font, fill=color)
    return img

def draw_polygon(draw, points, outline_color, width=3):
    """绘制多边形轮廓（支持指定线条宽度，保持不变）"""
    if len(points) < 3:
        return
    
    # 确保points是正确的格式
    if isinstance(points, np.ndarray):
        points = points.tolist()
    
    # 将点转换为tuple格式
    polygon_points = []
    for point in points:
        if isinstance(point, (list, tuple, np.ndarray)):
            polygon_points.append((int(point[0]), int(point[1])))
        else:
            print(f"Warning: Invalid point format: {point}")
            continue
    
    if len(polygon_points) >= 3:
        # 用line方法绘制带宽度的闭合轮廓
        line_points = polygon_points + [polygon_points[0]]  # 闭合多边形
        draw.line(line_points, fill=outline_color, width=width)

def draw_classification_points(draw, points, color, radius=4):
    """绘制分类用的特征点（保持不变）"""
    if not points:
        return
        
    for point in points:
        if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
            x, y = int(point[0]), int(point[1])
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=color, outline=(0, 0, 0), width=1)

for page in tqdm(annotations):
    img_basename = os.path.splitext(os.path.basename(page["image_path"]))[0]
    vis_img_fp = os.path.join(MASK_VIS_ROOT, img_basename+'_maskvis.png')
    out_img_fp = os.path.join(OUT_ROOT, img_basename+'_mask_shapevis.png')
    
    if not os.path.exists(vis_img_fp):
        print('[Skip] Visualization file not found:', vis_img_fp)
        continue
        
    try:
        image = Image.open(vis_img_fp).convert("RGB")
        draw = ImageDraw.Draw(image)
    except Exception as e:
        print(f"[Error] Failed to open image {vis_img_fp}: {e}")
        continue
    
    # ########### 处理Panel：统一颜色，根据classification_points绘制 ###########
    for i, frame in enumerate(page["frames"]):
        if "shape_type" in frame and frame["shape_type"] in panel_map:
            shape_type = frame["shape_type"]
            shape_label = panel_map.get(shape_type, "Unknown")
            
            # 1. 优先根据classification_points绘制panel轮廓
            if "classification_points" in frame and frame["classification_points"]:
                classification_points = frame["classification_points"]
                # 用统一颜色绘制panel多边形轮廓
                draw_polygon(draw, classification_points, PANEL_COLOR, width=4)
                
                # 绘制特征点（可选，用于展示分类依据）
                draw_classification_points(draw, classification_points, PANEL_COLOR, radius=3)
                
                # 计算中心点用于标注文字
                if len(classification_points) >= 3:
                    center_x = sum(p[0] for p in classification_points) / len(classification_points)
                    center_y = sum(p[1] for p in classification_points) / len(classification_points)
                    # 标注panel形状
                    text = f"Panel: {shape_label}"
                    image = draw_en_text(image, text, 
                                       (int(center_x - 80), int(center_y - 20)), 
                                       size=42, color=PANEL_COLOR)
            
            # 2. 若没有classification_points，fallback到four_points（可选）
            elif "four_points" in frame and frame["four_points"]:
            # if "four_points" in frame and frame["four_points"]:
                four_points = frame["four_points"]
                draw_polygon(draw, four_points, PANEL_COLOR, width=4)
                
                if len(four_points) >= 4:
                    center_x = sum(p[0] for p in four_points) / len(four_points)
                    center_y = sum(p[1] for p in four_points) / len(four_points)
                    text = f"Panel: {shape_label}"
                    image = draw_en_text(image, text, (int(center_x - 80), int(center_y - 20)), size=42, color=PANEL_COLOR)
            
            # 3. 若以上都没有，fallback到bbox（可选）
            elif "bbox" in frame:
                x1, y1, x2, y2 = frame["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline=PANEL_COLOR, width=4)
                cx = int((x1 + x2) // 2)
                cy = int((y1 + y2) // 2)
                text = f"Panel: {shape_label}"
                image = draw_en_text(image, text, (cx-80, cy-20), size=42, color=PANEL_COLOR)
        
        # ########### 处理Bubble：不绘制box，只标注形状 ###########
        if 'dialogs' in frame:
            for j, dialog in enumerate(frame["dialogs"]):
                if "bubble_type" in dialog and dialog["bubble_type"] in bubble_shape_map:
                    bubble_type = dialog["bubble_type"]
                    shape_label = bubble_shape_map.get(bubble_type, "Unknown")
                    
                    # 只标注形状文字，不绘制box
                    if "bbox" in dialog:  # 用bbox定位文字位置
                        dx1, dy1, dx2, dy2 = dialog["bbox"]
                        dcx = int((dx1 + dx2) // 2)
                        dcy = int(dy1 + 5)  # 靠上显示
                        text = f"Bubble: {shape_label}"
                        image = draw_en_text(image, text, (dcx-80, dcy), size=32, color=BUBBLE_COLOR)
    
    # ########### 绘制图例（简化，只保留panel统一颜色说明） ###########
    legend_y = 10
    # Panel图例
    image = draw_en_text(image, "Panel (统一颜色):", (10, legend_y), size=24, color=PANEL_COLOR)
    # 绘制颜色块
    draw.rectangle([10, legend_y+30, 30, legend_y+50], fill=PANEL_COLOR, outline=(0,0,0))
    
    # Bubble图例（只标文字）
    legend_y += 70
    image = draw_en_text(image, "Bubble (仅标注形状):", (10, legend_y), size=24, color=BUBBLE_COLOR)
    
    image.save(out_img_fp)
    # print(f"Saved: {out_img_fp}")

print("✅ 形状可视化完成！")