# utils/datasets.py
import torch
from typing import Dict, Any, List
import os
import json
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, ViTImageProcessor
import torch
from torch.utils.data import Dataset, Sampler, RandomSampler
from torchvision import transforms
from utils import get_bucket_size, resize_and_center_crop, get_relative_bbox, mask_dialogs_from_image


def image_transform(pil_image):
    fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return fn(pil_image)

def _norm_xyxy(b, W, H):
    x1, y1, x2, y2 = b
    return [x1/W, y1/H, x2/W, y2/H]

def _xyxy_to_cxcywh(xyxy):
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = (x2 - x1)
    h  = (y2 - y1)
    return [cx, cy, w, h]

def _offsets_from_four_points(four_points, bbox, W, H):
    x1, y1, x2, y2 = bbox
    base = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    offs = []
    scale = float(max(W, H))
    for (px, py), (bx, by) in zip(four_points, base):
        offs += [(px - bx)/scale, (py - by)/scale]
    return offs

def pad_to_max_tensor(tensor: torch.Tensor, max_len: int, pad_val: float = 0.0):
    """Pad 1D / 2D 张量到 max_len；保持维度不变"""
    if tensor.numel() == 0:
        if tensor.ndim == 1:
            return torch.full((max_len,), pad_val, dtype=tensor.dtype, device=tensor.device)
        else:
            return torch.full((max_len, tensor.shape[-1]), pad_val, dtype=tensor.dtype, device=tensor.device)
    n = tensor.shape[0]
    if n >= max_len:
        return tensor[:max_len]
    if tensor.ndim == 1:
        pad = torch.full((max_len - n,), pad_val, dtype=tensor.dtype, device=tensor.device)
    else:
        pad = torch.full((max_len - n, tensor.shape[1]), pad_val, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=0)

class MangaLayoutDataset(Dataset):
    def __init__(self, ann_source, image_dir=None, cfg=None,
                 max_panels=16, max_num_ips=4, max_num_ip_sources=1,
                 min_ip_height=10, min_ip_width=10, ip_flip_rate=0.5):
        self.cfg = cfg
        self.samples = []
        if os.path.isdir(ann_source):
            files = [os.path.join(ann_source, f) for f in os.listdir(ann_source) if f.endswith(".json")]
            files.sort()
            for p in files:
                with open(p, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                self.samples.append(ann)
        else:
            with open(ann_source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.samples = data
            elif isinstance(data, dict) and "frames" in data:
                self.samples = [data]
            elif isinstance(data, dict) and "annotations" in data:
                self.samples = data["annotations"]
            else:
                raise ValueError(f"Unsupported JSON format: {ann_source}")
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No annotations found from {ann_source}")

        # 图像处理器
        self.image_dir = image_dir
        self.max_panels = max_panels
        self.max_num_ips = max_num_ips
        self.max_num_ip_sources = max_num_ip_sources
        self.min_ip_height = min_ip_height
        self.min_ip_width = min_ip_width
        self.ip_flip_rate = ip_flip_rate
        self.clip_image_processor = CLIPImageProcessor()
    def __len__(self):
        return len(self.samples)
    def _get_page_image(self, ann):
        if self.image_dir is None: return None
        image_path = os.path.join(self.image_dir, ann["image_path"])
        try:
            page_img = Image.open(image_path).convert("RGB")
        except:
            page_img = Image.new("RGB", (224,224), (0,0,0))
        return page_img
    def _sample_ip_images(self, ann, frame):
        characters = frame.get("characters", [])
        ids = list({c["id"] for c in characters if "bbox" in c})[:self.max_num_ips]
        page_img = self._get_page_image(ann)
        ip_images = []
        ip_exists = []
        for char_id in ids:
            boxes = [c["bbox"] for c in characters if c["id"] == char_id]
            valid_boxes = [b for b in boxes if (b[3]-b[1])>=self.min_ip_height and (b[2]-b[0])>=self.min_ip_width]
            n = 0
            for b in valid_boxes[:self.max_num_ip_sources]:
                x1,y1,x2,y2 = b
                crop = page_img.crop([x1,y1,x2,y2]) if page_img else Image.new("RGB", (224,224), (0,0,0))
                if np.random.rand() < self.ip_flip_rate:
                    crop = ImageOps.mirror(crop)
                ip_images.append(crop)
                ip_exists.append(1); n+=1
            while n < self.max_num_ip_sources:
                ip_images.append(Image.new("RGB", (224,224), (0,0,0)))
                ip_exists.append(0); n+=1
        pad_to = self.max_num_ips * self.max_num_ip_sources
        while len(ip_images) < pad_to:
            ip_images.append(Image.new("RGB", (224,224), (0,0,0)))
            ip_exists.append(0)
        ip_img_tensor = self.clip_image_processor(images=ip_images, return_tensors="pt").pixel_values
        ip_exists_tensor = torch.tensor(ip_exists, dtype=torch.float32)
        return ip_img_tensor, ip_exists_tensor
    def __getitem__(self, idx):
        ann = self.samples[idx]
        width = ann["width"]
        height = ann["height"]
        style_vec = torch.tensor([
            ann["style_parameters"]["layout_density"],
            ann["style_parameters"]["alignment_score"],
            ann["style_parameters"]["shape_instability"],
            ann["style_parameters"]["breakout_intensity"]
        ], dtype=torch.float32)
        frames = ann.get("frames", [])
        out_panels = []
        for pi, frame in enumerate(frames):
            panel = {
                "panel_idx": pi,
                "bbox": frame["bbox"],
                "shape_type": frame.get("shape_type", "panel_rect"),
                "caption": frame.get("caption", ""),
                "four_points": frame.get("four_points"),
                "classification_points": frame.get("classification_points"),
                "characters": frame.get("characters", []),
                "dialogs": frame.get("dialogs", [])
            }
            ip_img_tensor, ip_exists_tensor = self._sample_ip_images(ann, frame)
            panel["ip_images"] = ip_img_tensor      # (max_num_ips*max_num_ip_sources, 3, 224, 224)
            panel["ip_exists"] = ip_exists_tensor   # (max_num_ips*max_num_ip_sources,)
            out_panels.append(panel)
        return {
            "width": width,
            "height": height,
            "style_vector": style_vec,
            "frames": out_panels,
        }
        
def collate_fn(batch: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    # 取各种长度上限
    max_elem_len = cfg["dataset"]["max_elements"]
    max_panels   = cfg["dataset"]["parameters"]["max_panels"]
    max_dialogs  = cfg["dataset"]["parameters"]["max_dialogs"]
    max_chars    = cfg["dataset"]["parameters"]["max_characters"]
    max_num_ips  = cfg["dataset"]["parameters"]["max_num_ips"]
    max_num_ip_sources = cfg["dataset"]["parameters"].get("max_num_ip_sources", 1)
    TYPE_PAD    = cfg["layout_types"]["TYPE_PAD"]
    TYPE_PAGE   = cfg["layout_types"]["TYPE_PAGE"]
    TYPE_PANEL  = cfg["layout_types"]["TYPE_PANEL"]
    TYPE_CHAR   = cfg["layout_types"]["TYPE_CHAR"]
    TYPE_DIALOG = cfg["layout_types"]["TYPE_DIALOG"]
    shape_map = {k: v["id"] for k, v in cfg["panel_shapes"].items()}
    shape_map_dialog = {"bubble_oval":0, "bubble_flower":1, "bubble_burst":2, "bubble_rect":3}

    bs = len(batch)
    # 各类属性临时存储
    style_vecs     = []
    element_types  = []
    element_indices= []
    parent_panel_idx=[]
    panel_captions = []
    panel_bboxes   = []
    panel_offsets  = []
    panel_classes  = []
    panel_ip_images= []
    panel_ip_exists= []
    dialog_bboxes  = []
    dialog_shapes  = []
    dialog_breakout_labels = []
    dialog_breakout_ratios = []
    character_bboxes    = []
    character_breakout_labels = []
    character_breakout_ratios = []
    width = []
    height = []

    for ann in batch:
        W, H = ann["width"], ann["height"]
        width.append(W)
        height.append(H)
        style_vecs.append(ann['style_vector'])
        frames = ann.get("frames", [])
        
        # token序列展平
        et, ei, parent_idx = [TYPE_PAGE],[0],[-1]
        panels, dialogs, chars = [], [], []
        pcaps = []
        pbxs, poffs, pcls, ip_imgs, ip_exists = [], [], [], [], []
        for pi, p in enumerate(frames):
            # 展panel token
            panels.append({"panel_idx": pi, "frame": p})
            et.append(TYPE_PANEL); ei.append(pi); parent_idx.append(-1)
            # panel属性
            bbox = p["bbox"]
            four = p.get("four_points")
            if four is None:
                x1, y1, x2, y2 = bbox
                four = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            xyxy_n = _norm_xyxy(bbox, W, H)
            cxcywh_n = _xyxy_to_cxcywh(xyxy_n)
            pbxs.append(cxcywh_n)
            poffs.append(_offsets_from_four_points(four, bbox, W, H))
            pcls.append(shape_map.get(p.get("shape_type", "panel_rect"), 0))
            pcaps.append(p.get("caption", ""))
            ip_imgs.append(p["ip_images"])
            ip_exists.append(p["ip_exists"])
            # 展dialogs/chars
            for d in p.get("dialogs", []):
                dialogs.append({"panel_idx": pi, "dialog": d})
            for c in p.get("characters", []):
                chars.append({"panel_idx": pi, "char": c})
        # dialog/char同步展开token及属性
        dbxs, dlabels, dratios, dshapes = [], [], [], []
        cbxs, clabels, cratios = [], [], []
        for j, d in enumerate(dialogs):
            et.append(TYPE_DIALOG); ei.append(j); parent_idx.append(d["panel_idx"])
            dg = d["dialog"]
            xyxy = dg.get("dialog_bbox") or dg.get("bbox") or [0, 0, 1, 1]
            x1,y1,x2,y2 = xyxy
            cx = (x1+x2)/2 / W
            cy = (y1+y2)/2 / H
            w  = (x2-x1) / W
            h  = (y2-y1) / H
            dbxs.append([cx,cy,w,h])
            br = float(dg.get("breakout_ratio", 0.0))
            dratios.append(br)
            dlabels.append(1 if br > 1e-6 else 0)
            dshapes.append(shape_map_dialog.get(dg.get("bubble_type", None), 0))
        for k, c in enumerate(chars):
            et.append(TYPE_CHAR); ei.append(k); parent_idx.append(c["panel_idx"])
            ch = c["char"]
            xyxy = ch.get("bbox") or [0,0,1,1]
            x1,y1,x2,y2 = xyxy
            cx = (x1+x2)/2 / W
            cy = (y1+y2)/2 / H
            w  = (x2-x1) / W
            h  = (y2-y1) / H
            cbxs.append([cx,cy,w,h])
            br = float(ch.get("breakout_ratio", 0.0))
            cratios.append(br)
            clabels.append(1 if br > 1e-6 else 0)

        # 加入到batch
        element_types.append(torch.tensor(et, dtype=torch.long))
        element_indices.append(torch.tensor(ei, dtype=torch.long))
        parent_panel_idx.append(torch.tensor(parent_idx, dtype=torch.long))
        panel_captions.append(pcaps)
        # pad panel属性
        panel_bboxes.append(torch.tensor(pbxs, dtype=torch.float32))
        panel_offsets.append(torch.tensor(poffs, dtype=torch.float32))
        panel_classes.append(torch.tensor(pcls, dtype=torch.long))
        # ip
        # pad至max_panels
        ipimg = ip_imgs + [torch.zeros_like(ip_imgs[0])] * (max_panels-len(ip_imgs)) if ip_imgs else \
            [torch.zeros((max_num_ips*max_num_ip_sources,3,224,224))]*max_panels
        ipex = ip_exists + [torch.zeros_like(ip_exists[0])] * (max_panels-len(ip_exists)) if ip_exists else \
            [torch.zeros(max_num_ips*max_num_ip_sources)]*max_panels
        panel_ip_images.append(torch.stack(ipimg))
        panel_ip_exists.append(torch.stack(ipex))
        # pad panel_captions
        if len(pcaps)<max_panels: pcaps += [""]*(max_panels-len(pcaps))
        # pad dialog/char属性
        dialog_bboxes.append(pad_to_max_tensor(dbxs, max_dialogs))
        dialog_shapes.append(pad_to_max_tensor(dshapes, max_dialogs, pad_val=-1))
        dialog_breakout_labels.append(pad_to_max_tensor(dlabels, max_dialogs, pad_val=0))
        dialog_breakout_ratios.append(pad_to_max_tensor(dratios, max_dialogs, pad_val=0.0))
        character_bboxes.append(pad_to_max_tensor(cbxs, max_chars))
        character_breakout_labels.append(pad_to_max_tensor(clabels, max_chars, pad_val=0))
        character_breakout_ratios.append(pad_to_max_tensor(cratios, max_chars, pad_val=0.0))

    # pad所有token序列（batch组）
    element_types = torch.stack([
        pad_to_max_tensor(et, max_elem_len, pad_val=TYPE_PAD) for et in element_types
    ])
    element_indices = torch.stack([
        pad_to_max_tensor(ei, max_elem_len, pad_val=0) for ei in element_indices
    ])
    parent_panel_indices = torch.stack([
        pad_to_max_tensor(pi, max_elem_len, pad_val=-1) for pi in parent_panel_idx
    ])
    token_panel_mask     = (element_types == TYPE_PANEL)
    token_dialog_mask    = (element_types == TYPE_DIALOG)
    token_character_mask = (element_types == TYPE_CHAR)
    panel_bboxes = torch.stack([pad_to_max_tensor(pb, max_panels) for pb in panel_bboxes])
    panel_offsets= torch.stack([pad_to_max_tensor(po, max_panels) for po in panel_offsets])
    panel_classes= torch.stack([pad_to_max_tensor(pc, max_panels, pad_val=-1) for pc in panel_classes])
    panel_ip_images = torch.stack(panel_ip_images)
    panel_ip_exists = torch.stack(panel_ip_exists)
    # dialogs/chars
    dialog_bboxes = torch.stack(dialog_bboxes)
    dialog_shapes = torch.stack(dialog_shapes)
    dialog_breakout_labels = torch.stack(dialog_breakout_labels)
    dialog_breakout_ratios = torch.stack(dialog_breakout_ratios)
    character_bboxes = torch.stack(character_bboxes)
    character_breakout_labels = torch.stack(character_breakout_labels)
    character_breakout_ratios = torch.stack(character_breakout_ratios)
    # 固定长度mask
    panel_mask = torch.stack([
        torch.arange(max_panels) < len(b.get("frames", []))
        for b in batch
    ])
    dialog_mask = torch.stack([
        torch.arange(max_dialogs) < sum(len(fr.get("dialogs",[])) for fr in b.get("frames",[]))
        for b in batch
    ])
    character_mask = torch.stack([
        torch.arange(max_chars) < sum(len(fr.get("characters",[])) for fr in b.get("frames",[]))
        for b in batch
    ])
    # 父索引
    dialog_parent_idx = torch.full((bs, max_dialogs), -1, dtype=torch.long)
    char_parent_idx   = torch.full((bs, max_chars), -1, dtype=torch.long)
    for b_idx, (et, parent_idx) in enumerate(zip(element_types, parent_panel_indices)):
        d_idxs = torch.nonzero(et == TYPE_DIALOG, as_tuple=False).squeeze(-1)
        for i, di in enumerate(d_idxs):
            if i < max_dialogs:
                dialog_parent_idx[b_idx, i] = parent_idx[di]
        c_idxs = torch.nonzero(et == TYPE_CHAR, as_tuple=False).squeeze(-1)
        for i, ci in enumerate(c_idxs):
            if i < max_chars:
                char_parent_idx[b_idx, i] = parent_idx[ci]
    # style
    style_vector = torch.stack(style_vecs)
    width = torch.tensor(width, dtype=torch.int64)
    height= torch.tensor(height, dtype=torch.int64)
    return {
        "width": width,
        "height": height,
        "style_vector": style_vector,
        "element_types": element_types,
        "element_indices": element_indices,
        "parent_panel_indices": parent_panel_indices,
        "panel_captions": panel_captions,
        "panel_bboxes": panel_bboxes,
        "panel_offsets": panel_offsets,
        "panel_classes": panel_classes,
        "panel_ip_images": panel_ip_images,   # (B, max_panels, ip_dim, 3, 224, 224)
        "panel_ip_exists": panel_ip_exists,   # (B, max_panels, ip_dim)
        "dialog_bboxes": dialog_bboxes,
        "dialog_shapes": dialog_shapes,
        "dialog_breakout_labels": dialog_breakout_labels,
        "dialog_breakout_ratios": dialog_breakout_ratios,
        "character_bboxes": character_bboxes,
        "character_breakout_labels": character_breakout_labels,
        "character_breakout_ratios": character_breakout_ratios,
        "token_panel_mask": token_panel_mask,
        "token_dialog_mask": token_dialog_mask,
        "token_character_mask": token_character_mask,
        "panel_mask": panel_mask,
        "dialog_mask": dialog_mask,
        "character_mask": character_mask,
        "dialog_parent_idx": dialog_parent_idx,
        "char_parent_idx": char_parent_idx,
    }