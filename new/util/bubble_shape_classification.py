import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils
import cv2

def decode_mask(mask_rle, image_shape):
    mask = mask_utils.decode(mask_rle)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    assert mask.shape == image_shape[:2], f"Mask shape mismatch: mask {mask.shape}, img {image_shape}"
    return mask.astype(np.uint8)

def analyze_shape_and_four_points(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area == 0 or perimeter == 0:
        return None, None, None
    bbox = cv2.boundingRect(contour)
    roundness = 4 * np.pi * area / (perimeter ** 2)
    rectangularity = area / (bbox[2] * bbox[3])
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corner_count = len(approx)
    compactness = perimeter ** 2 / area
    aspect_ratio = bbox[3] / bbox[2] if bbox[2] != 0 else 0
    features = {
        'roundness': roundness,
        'rectangularity': rectangularity,
        'convexity': convexity,
        'corners': corner_count,
        'compactness': compactness,
        'aspect_ratio': aspect_ratio,
        'irregularity': 1 - min(roundness, rectangularity)
    }
    return features, None, None

def classify_bubble_shape(features):
    if features['rectangularity'] > 0.8 and 4 <= features['corners'] <= 5:
        return 'bubble_rect'
    if features['roundness'] > 0.68 and features['corners'] <= 7 and features['aspect_ratio'] <= 2.2:
        return 'bubble_oval'
    if features['corners'] >= 10 and features['irregularity'] > 0.39 and features['convexity'] < 0.93:
        return 'bubble_burst'
    if features['corners'] >= 8 and features['convexity'] > 0.90 and 0.5 < features['roundness'] < 0.72:
        return 'bubble_flower'
    return 'bubble_irregular'

ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_mask.json"
IMG_ROOT = "/data/DiffSensei-main/checkpoints/mangazero/images"
OUT_ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_shape.json"

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

for page in tqdm(annotations):
    img_fp = os.path.join(IMG_ROOT, page["image_path"])
    try:
        image = np.array(Image.open(img_fp).convert("RGB"))
        H, W = image.shape[:2]
    except:
        H, W = page["frames"][0]["mask_rle"]["size"] if "mask_rle" in page["frames"][0] else (None, None)
        if H is None:
            continue

    for i, frame in enumerate(page["frames"]):
        if 'dialogs' in frame:
            for j, dialog in enumerate(frame['dialogs']):
                if 'mask_rle' in dialog and dialog['mask_rle'] is not None:
                    try:
                        mask = decode_mask(dialog['mask_rle'], (H, W))
                        features, _, _ = analyze_shape_and_four_points(mask)
                        if features:
                            bubble_type = classify_bubble_shape(features)
                            dialog['bubble_type'] = bubble_type
                            dialog['shape_features'] = features
                    except Exception as e:
                        print(f"[bubble shape分析失败]: {page['image_path']} frame{i}-dialog{j}: {e}")

with open(OUT_ANN_FILE, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)