import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
from segment_anything import SamPredictor
from build_sam import build_sam2
# from sam2.build_sam import build_sam2
import numpy as np
from pycocotools import mask as mask_utils

# [1] 设备选择
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# [2] 加载SAM2
sam2_checkpoint = "/data/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "/data/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
predictor = SamPredictor(sam2)

# [3] mask转rle
from pycocotools import mask as mask_utils
def mask_to_rle_str(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle['counts'] = rle['counts'].decode()
    return rle

# [4] 主循环: 载annotation和图像，panel/dialog分割
with open('/data/DiffSensei-main/checkpoints/mangazero/filtered_annotations.json', 'r', encoding='utf-8') as f:
    annotations = json.load(f)

img_root = "/data/DiffSensei-main/checkpoints/mangazero/images"

for page in tqdm(annotations):
    img_fp = os.path.join(img_root, page["image_path"])
    image = np.array(Image.open(img_fp).convert("RGB"))
    predictor.set_image(image)        # 只set一次即可

    for i, frame in enumerate(page["frames"]):
        # Panel分割
        panel_box = np.array(frame["bbox"])[None, :]  # shape: (1,4)
        masks, _, _ = predictor.predict(box=panel_box, multimask_output=False)
        mask = masks[0]
        frame['mask_rle'] = mask_to_rle_str(mask)     # 直接写成rle到dict

        # Dialog分割
        for j, dialog in enumerate(frame.get("dialogs", [])):
            d_box = np.array(dialog["bbox"])[None, :]
            d_masks, _, _ = predictor.predict(box=d_box, multimask_output=False)
            d_mask = d_masks[0]
            dialog['mask_rle'] = mask_to_rle_str(d_mask)

with open("/data/DiffSensei-main/checkpoints/mangazero/filter_annotations_mask.json", "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)