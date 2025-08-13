import os
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

def overlay_mask(image, mask, color, alpha=0.4):
    out = image.astype(np.float32).copy()
    color255 = np.array(color) * 255
    out[mask] = (1 - alpha) * out[mask] + alpha * color255
    return out.astype(np.uint8)

IMG_ROOT = '/data/DiffSensei-main/checkpoints/mangazero/images'
# ANN_FILE = '/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_mask.json'
# ANN_FILE = '/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_mask_refine.json'
ANN_FILE = '/data/DiffSensei-main/checkpoints/mangazero/f_annotations_detect.json'
# OUT_DIR = '/data/DiffSensei-main/checkpoints/mangazero/image_mask_vis'
OUT_DIR = '/data/DiffSensei-main/checkpoints/mangazero/image_mask_detect'
os.makedirs(OUT_DIR, exist_ok=True)

color_map = {'panel': (1, 0, 0), 'dialog': (0, 1, 0)}  # 红/绿
alpha = 0.4

import json
with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

for page in tqdm(annotations):
    # 1. 使用annotation中的全相对路径
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
    for i, frame in enumerate(page['frames']):
        # panel
        if 'mask_rle' in frame and frame['mask_rle'] is not None:
            mask = mask_utils.decode(frame['mask_rle']).astype(bool)
            if mask.shape != image.shape[:2]:
                print(f"[警告] Panel mask尺寸不符: {img_path}, frame {i}, mask {mask.shape}, image {image.shape}")
                continue
            vis_img = overlay_mask(vis_img, mask, color_map['panel'], alpha)
        # dialogs
        for j, dialog in enumerate(frame.get('dialogs', [])):
            if 'mask_rle' in dialog and dialog['mask_rle'] is not None:
                mask = mask_utils.decode(dialog['mask_rle']).astype(bool)
                if mask.shape != image.shape[:2]:
                    print(f"[警告] Dialog mask尺寸不符: {img_path}, frame {i}-dialog{j}, mask {mask.shape}, image {image.shape}")
                    continue
                vis_img = overlay_mask(vis_img, mask, color_map['dialog'], alpha)

    out_img_name = os.path.splitext(os.path.basename(page["image_path"]))[0] + '_maskvis.png'
    save_fp = os.path.join(OUT_DIR, out_img_name)
    Image.fromarray(vis_img).save(save_fp)