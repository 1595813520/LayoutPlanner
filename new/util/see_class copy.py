import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from tqdm import tqdm

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
    'panel_polygon': "Polygon",
    'panel_irregular': "Irregular", 
    'panel_parallelogram': "parallelogram",
}

IMG_ROOT = '/data/DiffSensei-main/checkpoints/mangazero/images'
# ANN_FILE = '/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_shape.json'
ANN_FILE = '/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_stable_shape.json'
MASK_VIS_ROOT = '/data/DiffSensei-main/checkpoints/mangazero/image_mask_vis_refine'
OUT_ROOT = '/data/DiffSensei-main/checkpoints/mangazero/image_mask_with_stable_shape'
os.makedirs(OUT_ROOT, exist_ok=True)

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

def draw_en_text(img, text, xy, size=36, color=(255,255,0)):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=color, stroke_fill="black", stroke_width=2)
    return img

for page in tqdm(annotations):
    img_basename = os.path.splitext(os.path.basename(page["image_path"]))[0]
    vis_img_fp = os.path.join(MASK_VIS_ROOT, img_basename+'_maskvis.png')
    out_img_fp = os.path.join(OUT_ROOT, img_basename+'_mask_shapevis.png')
    
    if not os.path.exists(vis_img_fp):
        print('[Skip] Visualization file not found:', vis_img_fp)
        continue
        
    try:
        image = Image.open(vis_img_fp).convert("RGB")
        H, W = image.size
    except Exception as e:
        print(f"[Error] Failed to open image {vis_img_fp}: {e}")
        continue
    
    # Label panels
    for i, frame in enumerate(page["frames"]):
        if "shape_type" in frame and "bbox" in frame and frame["shape_type"] in panel_map:
            x1, y1, x2, y2 = frame["bbox"]
            cx = int((x1 + x2) // 2)
            cy = int(y1 + 4)
            shape_label = panel_map.get(frame["shape_type"], "Unknown")
            image = draw_en_text(image, f"Panel: {shape_label}", (cx-60, cy), size=36, color=(255,210,0))
        
        # Label bubbles with new shape classification
        if 'dialogs' in frame:
            for j, dialog in enumerate(frame["dialogs"]):
                if "bubble_type" in dialog and "bbox" in dialog and dialog["bubble_type"] in bubble_shape_map:
                    dx1, dy1, dx2, dy2 = dialog["bbox"]
                    dcx = int((dx1 + dx2) // 2)
                    dcy = int(dy1 + 2)
                    shape_type = bubble_shape_map.get(dialog["bubble_type"], "Unknown")
                    image = draw_en_text(image, f"Bubble: {shape_type}", (dcx-70, dcy), size=32, color=(70,200,255))
    
    image.save(out_img_fp)