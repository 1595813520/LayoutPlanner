import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from pycocotools import mask as mask_utils

ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/filter_annotations.json"
IMG_ROOT = "/data/DiffSensei-main/checkpoints/mangazero/images"
OUT_ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_detect.json"

def mask_to_rle(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    if isinstance(rle, list):
        rle = rle[0]
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode()
    return rle

def expand_box(x1, y1, w, h, H, W, ratio=0.04):
    # ratio=0.04即扩展4%，可自行调整
    expand_x = int(w * ratio)
    expand_y = int(h * ratio)
    nx1 = max(0, x1 - expand_x)
    ny1 = max(0, y1 - expand_y)
    nx2 = min(W, x1 + w + expand_x)
    ny2 = min(H, y1 + h + expand_y)
    nwidth = nx2 - nx1
    nheight = ny2 - ny1
    return nx1, ny1, nwidth, nheight

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

for page in tqdm(annotations):
    img_path = os.path.join(IMG_ROOT, page["image_path"])
    try:
        image = np.array(Image.open(img_path).convert("RGB"))
        H, W = image.shape[:2]
    except:
        print(f"[图片缺失] {page['image_path']}")
        continue

    for fi, frame in enumerate(page.get("frames", [])):
        bbox = frame.get("bbox", None)
        if bbox is None: continue
        x1, y1, w, h = [int(r) for r in bbox]
        # === 扩大panel box ===
        ex, ey, ew, eh = expand_box(x1, y1, w, h, H, W, ratio=0.04)
        ex2, ey2 = ex + ew, ey + eh

        # 1. 截取扩大的ROI区域
        roi_img = image[ey:ey2, ex:ex2, :]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)

        # 2. 二值化（根据你数据调节）
        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 35, 15
        )
        # 3. 闭操作
        kernel = np.ones((11,11), np.uint8)
        bw_close = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        # 4. 最大连通域
        contours, _ = cv2.findContours(bw_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        panel_cnt = max(contours, key=cv2.contourArea)
        # 偏移回整图坐标
        panel_cnt_full = panel_cnt + np.array([[[ex, ey]]])

        # 5. 生成全图mask
        panel_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(panel_mask, [panel_cnt_full], -1, color=1, thickness=-1)

        # 6. 多边形点平滑
        epsilon = 0.012 * cv2.arcLength(panel_cnt, True)
        approx = cv2.approxPolyDP(panel_cnt, epsilon, True) + np.array([[[ex, ey]]])
        approx_pts = approx.reshape(-1, 2).tolist()

        # 7. 存入annotation
        frame["mask_rle"] = mask_to_rle(panel_mask)
        frame["polygon_pts"] = approx_pts

with open(OUT_ANN_FILE, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)

print("已保存:", OUT_ANN_FILE)