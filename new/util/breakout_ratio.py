import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, box as shapely_box
from pycocotools import mask as mask_utils

ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_shape.json"
OUT_ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_breakout.json"

def decode_mask(mask_rle, image_shape):
    mask = mask_utils.decode(mask_rle)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    assert mask.shape == image_shape[:2], f"Mask shape mismatch: mask {mask.shape}, img {image_shape}"
    return mask.astype(np.uint8)

def compute_breakout_ratio_box(char_box, panel_box):
    xA, yA, xB, yB = char_box
    xC, yC, xD, yD = panel_box
    ix1 = max(xA, xC)
    iy1 = max(yA, yC)
    ix2 = min(xB, xD)
    iy2 = min(yB, yD)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih
    char_area = max(1, (xB - xA) * (yB - yA))
    ratio = 1 - inter_area / char_area
    return max(0.0, min(1.0, ratio))

def compute_breakout_ratio_mask(char_box, panel_mask):
    xA, yA, xB, yB = map(int, char_box)
    H, W = panel_mask.shape
    xA, yA = max(0, xA), max(0, yA)
    xB, yB = min(W, xB), min(H, yB)
    if xA >= xB or yA >= yB:
        return 1.0
    char_area = max(1, (xB - xA) * (yB - yA))
    mask_roi = panel_mask[yA:yB, xA:xB]
    inside_pixel = np.count_nonzero(mask_roi)
    ratio = 1 - inside_pixel / char_area
    return max(0.0, min(1.0, ratio))

def analyze_breakout(annotations, mode='mask'):
    for page in tqdm(annotations):
        for fi, frame in enumerate(page["frames"]):
            breakout_elements = []
            # --- 获取对应panel-区域定义 ---
            H = W = None
            if 'mask_rle' in frame and frame['mask_rle'] is not None:
                # Decode mask尺寸
                try:
                    H,W = frame["mask_rle"]["size"]
                except:
                    pass
            panel_box = frame.get("bbox", None)
            panel_pts = frame.get("four_points", None)
            panel_mask = None
            if mode == 'mask' and 'mask_rle' in frame and frame['mask_rle'] is not None and H is not None and W is not None:
                panel_mask = decode_mask(frame['mask_rle'], (H, W))
            if "characters" in frame:
                for char in frame["characters"]:
                    char_box = char.get("bbox", None)
                    cid = char.get("id", None)
                    if char_box is None or cid is None:
                        continue
                    # 三种方式任选
                    if mode == 'box':
                        if panel_box is None: continue
                        ratio = compute_breakout_ratio_box(char_box, panel_box)
                    elif mode == 'mask':
                        if panel_mask is None: continue
                        ratio = compute_breakout_ratio_mask(char_box, panel_mask)
                    else:
                        raise ValueError('Invalid mode: choose from "box", "mask"')
                    char["breakout_ratio"] = ratio
                    if ratio > 0.01:
                        breakout_elements.append({
                            "element_id": cid,
                            "breakout_area": round(ratio, 4)
                        })
            if breakout_elements:
                frame["breakout_elements"] = breakout_elements

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# 可选 mode: 'box' | 'mask' | 'polygon'
# analyze_breakout(annotations, mode='box')     # 矩形
analyze_breakout(annotations, mode='mask')  # mask像素


with open(OUT_ANN_FILE, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)