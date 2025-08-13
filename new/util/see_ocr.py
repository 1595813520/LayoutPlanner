import os
import json
from PIL import Image, ImageDraw, ImageFont

ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_ocr.json"
IMG_ROOT = "/data/DiffSensei-main/checkpoints/mangazero/images"
OUT_IMG_DIR = "/data/DiffSensei-main/checkpoints/mangazero/scene_text_viz"
os.makedirs(OUT_IMG_DIR, exist_ok=True)

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

def visualize_scene_texts_save(annotations, IMG_ROOT, OUT_IMG_DIR, max_imgs=30):
    count = 0
    for page in annotations:
        img_fp = os.path.join(IMG_ROOT, page["image_path"])
        try:
            image = Image.open(img_fp).convert("RGB")
        except Exception as e:
            continue
        draw = ImageDraw.Draw(image)
        show_texts = []
        for frame in page.get("frames", []):
            for dialog in frame.get("dialogs", []):
                if dialog.get("shape_type") == "scene_texts":
                    bbox = dialog.get("bbox", None)
                    if bbox is None or len(bbox) != 4 or bbox == [0,0,0,0]:
                        continue
                    draw.rectangle(bbox, outline=(0,255,0), width=3)
                    show_texts.append(((bbox[0], bbox[1]), dialog.get('text', '')))
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = None
        for pos, txt in show_texts:
            if font:
                draw.text((pos[0], max(0,pos[1]-20)), txt, fill=(0,255,0), font=font)
            else:
                draw.text((pos[0], max(0,pos[1]-20)), txt, fill=(0,255,0))
        # 保存图片
        img_name = os.path.basename(page["image_path"])
        out_fp = os.path.join(OUT_IMG_DIR, f"scene_text_{count+1:03d}_{img_name}")
        image.save(out_fp)
        print(f"Saved: {out_fp}")
        count += 1
        if count >= max_imgs:
            break

visualize_scene_texts_save(annotations, IMG_ROOT, OUT_IMG_DIR, max_imgs=30)