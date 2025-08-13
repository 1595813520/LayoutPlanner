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

def polygon_angles(pts):
    angles = []
    n = len(pts)
    for i in range(n):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v1 = np.array(p0) - np.array(p1)
        v2 = np.array(p2) - np.array(p1)
        ang = np.arccos(
            np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0)
        )
        angles.append(np.degrees(ang))
    return angles

def is_parallel(v1, v2, tol=17):
    angle = np.degrees(
        np.arccos(
            np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0)
        )
    )
    return (abs(angle) < tol or abs(angle - 180) < tol)

def points_close(p1, p2, thres=16):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < thres

def fit_four_points(contour):
    """若约简点数>4, 使用最小外接四边形。"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.045 * peri, True)   # 适度放宽，降低点数
    pts = approx.reshape(-1, 2)
    if len(pts) == 3:
        # 补成4点
        pts = np.concatenate([pts, pts[2:3]], 0)
    elif len(pts) == 4:
        pass
    elif len(pts) > 4:
        # 用最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        pts = np.int0(box)
    elif len(pts) < 3:
        # 严重瑕疵直接用bbox
        x, y, w, h = cv2.boundingRect(contour)
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return pts

def classify_panel_shape_by_pts(pts):
    n = len(pts)
    pts = np.array(pts)
    # --- 1. 三角形 ---
    if n == 3 or (n == 4 and (points_close(pts[0], pts[1]) or points_close(pts[0], pts[2]) or points_close(pts[0], pts[3]) or points_close(pts[1], pts[2]) or points_close(pts[1], pts[3]) or points_close(pts[2], pts[3]))):
        return 'panel_triangle'
    if n == 4:
        v = [pts[(i+1)%4] - pts[i] for i in range(4)]
        lens = [np.linalg.norm(vec) for vec in v]
        angles = polygon_angles(pts)
        # -- Rect --
        is_all_90 = all(abs(a-90) < 23 for a in angles)
        is_opposite_equal = abs(lens[0] - lens[2])/max(lens[0], lens[2]) < 0.26 and abs(lens[1] - lens[3])/max(lens[1], lens[3]) < 0.26
        is_opp_parallel = is_parallel(v[0], v[2], tol=17) and is_parallel(v[1], v[3], tol=17)
        if is_all_90 and is_opposite_equal and is_opp_parallel:
            return 'panel_rect'
        # -- Parallelogram --
        if is_opp_parallel and is_opposite_equal and not is_all_90:
            return 'panel_parallelogram'
        # -- Trapezoid --
        parallel_01 = is_parallel(v[0], v[2], tol=17)
        parallel_12 = is_parallel(v[1], v[3], tol=17)
        if (parallel_01 != parallel_12):
            return 'panel_trapezoid'
        # 其余都勉强归为不规则四边形（但还能用四点拟合，不是很烂）
        return 'panel_irregular'
    if n > 4:
        # 超过4边但能被外接四边形包裹，也算irregular
        return "panel_irregular"
    # n<3
    return "panel_irregular"

# 主流程
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
        H, W = page["frames"][0].get("mask_rle", {}).get("size", (None, None))
        if H is None:
            continue

    for i, frame in enumerate(page["frames"]):
        if 'mask_rle' in frame and frame['mask_rle'] is not None:
            try:
                mask = decode_mask(frame['mask_rle'], (H, W))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                pts = fit_four_points(contour)

                shape_type = classify_panel_shape_by_pts(pts)
                frame['shape_type'] = shape_type
                frame['four_points'] = pts[:4].tolist()
            except Exception as e:
                print(f"[panel shape分析失败]: {page['image_path']} frame{i}: {e}")

with open(OUT_ANN_FILE, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)