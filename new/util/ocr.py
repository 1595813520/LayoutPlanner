import os
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import easyocr

def box_to_rect(bbox):
    # bbox: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    if isinstance(bbox, (list, np.ndarray)) and len(bbox) == 4:
        pt1 = bbox[0]
        pt3 = bbox[2]
        return [int(round(pt1[0])), int(round(pt1[1])), int(round(pt3[0])), int(round(pt3[1]))]
    else:
        print(f'[警告] bbox格式错误: {bbox}')
        return [0, 0, 0, 0]

def is_center_in_any_box(rect, all_boxes):
    """
    rect: [x1, y1, x2, y2]
    all_boxes: list of [x1, y1, x2, y2]
    return: True if center of rect is in any box
    """
    xc = (rect[0] + rect[2]) // 2
    yc = (rect[1] + rect[3]) // 2
    for box in all_boxes:
        if (box[0] <= xc <= box[2]) and (box[1] <= yc <= box[3]):
            return True
    return False

def analyze_scene_texts(annotations, IMG_ROOT):
    reader = easyocr.Reader(['en'], gpu=True)
    for page in tqdm(annotations):
        img_fp = os.path.join(IMG_ROOT, page["image_path"])
        try:
            image = np.array(Image.open(img_fp).convert("RGB"))
        except:
            continue
        # 收集所有dialog（气泡）bbox
        all_dialog_boxes = []
        for frame in page.get("frames", []):
            for dialog in frame.get("dialogs", []):
                b = dialog.get("bbox", None)
                if b is not None:
                    all_dialog_boxes.append(b)
        # EasyOCR全页面检测
        results = reader.readtext(image)
        scene_text_dialogs = []
        for item in results:
            bbox = item[0]
            rect = box_to_rect(bbox)
            if rect == [0,0,0,0]:
                continue
            # 用中心点判定是否落入了任何气泡框，如是则过滤掉
            if is_center_in_any_box(rect, all_dialog_boxes):
                continue
            # 保留scene_texts
            scene_text_dialogs.append({
                "bbox": rect,
                "shape_type": "scene_texts"
            })

        # 可自选加入策略，这里继续添加到第1个frame
        if scene_text_dialogs:
            if len(page.get("frames", [])) > 0:
                if 'dialogs' not in page["frames"][0]:
                    page["frames"][0]["dialogs"] = []
                page["frames"][0]["dialogs"].extend(scene_text_dialogs)

# ==== 主程序 ====
ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_shape_refine.json"
IMG_ROOT = "/data/DiffSensei-main/checkpoints/mangazero/images"
OUT_ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_ocr.json"

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

analyze_scene_texts(annotations, IMG_ROOT)

with open(OUT_ANN_FILE, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)