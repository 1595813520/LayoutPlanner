import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from tqdm import tqdm

def overlay_mask(image, mask, color, alpha=0.4):
    """将mask叠加到图像上"""
    out = image.astype(np.float32).copy()
    color255 = np.array(color) * 255
    out[mask] = (1 - alpha) * out[mask] + alpha * color255
    return out.astype(np.uint8)

def get_font(size=36):
    """尝试加载更好的字体，如果失败则使用默认字体"""
    try:
        # 尝试加载系统字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except:
            # 如果都失败，使用默认字体
            font = ImageFont.load_default()
    return font

def draw_boxes_with_index(image, boxes, color=(255, 255, 0), width=3, text_color=(255, 0, 255)):
    """在图像上绘制边界框并标注序号"""
    # 转换为PIL图像以便绘制
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    
    size = 40
    font = get_font(size)

    for idx, box in enumerate(boxes, start=1):  # 从1开始编号
        # 确保box是[x1, y1, x2, y2]格式
        x1, y1, x2, y2 = map(int, box)
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        # 在框的左上角绘制序号（稍微偏移，避免与边框重叠）
        text_pos = (x1 + 5, y1 + 5)  # 左上角偏移5像素
        draw.text(text_pos, str(idx), font=font, fill=text_color)
    
    # 转回numpy数组
    return np.array(img_pil)

IMG_ROOT = '/data/DiffSensei-main/checkpoints/mangazero/images'
ANN_FILE = '/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_mask_refine.json'
OUT_DIR = '/data/DiffSensei-main/checkpoints/mangazero/image_mask_vis_refine'
os.makedirs(OUT_DIR, exist_ok=True)

# 颜色配置：mask用半透明色，框用黄色，序号用紫色
color_map = {
    'panel': (1, 0, 0),        # 红色mask
    'dialog': (0, 1, 0),       # 绿色mask
    'panel_box':  (204, 204, 0),# 黄色框
    'index_text':  (204, 204, 0)
}
alpha = 0.4
box_width = 3  # 框的线宽

import json
with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

for page in tqdm(annotations):
    # 1. 构建图片路径
    img_path = os.path.join(IMG_ROOT, page["image_path"])
    # 2. 文件不存在时跳过
    if not os.path.exists(img_path):
        print(f"[警告] 图片文件不存在，已跳过: {img_path}")
        continue
    try:
        image = np.array(Image.open(img_path).convert("RGB"))
    except Exception as e:
        print(f"[警告] 读取图片失败 {img_path}: {e}")
        continue

    vis_img = image.copy()
    panel_boxes = []  # 收集所有panel的边界框
    
    for i, frame in enumerate(page['frames']):
        # 收集panel的边界框 (假设frame中有'bbox'字段，格式为[x1, y1, x2, y2])
        if 'bbox' in frame:
            panel_boxes.append(frame['bbox'])
        
        # 绘制panel的mask
        if 'mask_rle' in frame and frame['mask_rle'] is not None:
            mask = mask_utils.decode(frame['mask_rle']).astype(bool)
            if mask.shape != image.shape[:2]:
                print(f"[警告] Panel mask尺寸不符: {img_path}, frame {i}, mask {mask.shape}, image {image.shape}")
                continue
            vis_img = overlay_mask(vis_img, mask, color_map['panel'], alpha)
        
        # 绘制dialog的mask
        for j, dialog in enumerate(frame.get('dialogs', [])):
            if 'mask_rle' in dialog and dialog['mask_rle'] is not None:
                mask = mask_utils.decode(dialog['mask_rle']).astype(bool)
                if mask.shape != image.shape[:2]:
                    print(f"[警告] Dialog mask尺寸不符: {img_path}, frame {i}-dialog{j}, mask {mask.shape}, image {image.shape}")
                    continue
                vis_img = overlay_mask(vis_img, mask, color_map['dialog'], alpha)
    
    # 绘制所有panel的边界框及序号
    if panel_boxes:
        vis_img = draw_boxes_with_index(
            vis_img, 
            panel_boxes, 
            color=color_map['panel_box'], 
            width=box_width,
            text_color=color_map['index_text']
        )
    
    # 保存结果
    out_img_name = os.path.splitext(os.path.basename(page["image_path"]))[0] + '_maskvis.png'
    save_fp = os.path.join(OUT_DIR, out_img_name)
    Image.fromarray(vis_img).save(save_fp)