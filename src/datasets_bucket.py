import torch
from typing import Dict, Any, List
import os
import json
from torch.utils.data import Dataset, Sampler, RandomSampler
import random
import numpy as np
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor, ViTImageProcessor
from torchvision import transforms
from src.utils import get_bucket_size, resize_and_center_crop
from collections import defaultdict

def image_transform(pil_image):
    # 将PIL Image转换为Tensor，并归一化到[-1,1]
    fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    return fn(pil_image)

def _norm_xyxy(b, W, H):
    # 归一化到[0,1]
    x1, y1, x2, y2 = b
    if x2 < x1:  # swap or clamp
        x2 = x1 + 1e-6
    if y2 < y1:
        y2 = y1 + 1e-6
    return [
        max(0.0, min(1.0, x1 / W)),
        max(0.0, min(1.0, y1 / H)),
        max(0.0, min(1.0, x2 / W)),
        max(0.0, min(1.0, y2 / H)),
    ]

def _xyxy_to_cxcywh(xyxy):
    # (x1,y1,x2,y2) -> (cx,cy,w,h)
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return [cx, cy, w, h]

# 以 bbox 作为基准，计算 four_points 相对于 bbox 的偏移量
def _offsets_from_four_points(four_points, bbox, W, H):
    # panel的形变offsets
    x1, y1, x2, y2 = bbox
    base = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    offs = []
    scale = float(max(W, H))
    for (px, py), (bx, by) in zip(four_points, base):
        offs += [(px - bx)/scale, (py - by)/scale]
    return offs
'''
有几种类型需要 pad：
固定列数的二维数据
例：panel_bboxes（max_panels, 4）、offsets（max_panels, 8）、角色 bbox、对话框 bbox 等
⇒ 应该用类似 pad_to_max_tensor_box() 一样的接口，强制 target_dim
一维数据
例：分类 id 列表、mask 序列等
⇒ 可以用 pad_to_max_tensor_int() / pad_to_max_tensor_scalar() 这样的接口
纯 token 序列
例：element_types、element_indices
⇒ 可以用原来的 pad_to_max_tensor()，但只处理一维
'''
def pad_to_max_tensor(tensor, max_len, pad_val=0.0, target_dim=None):
    arr = torch.as_tensor(tensor)
    if arr.numel() == 0:
        if target_dim is not None and target_dim > 1:
            return torch.full((max_len, target_dim), pad_val, dtype=torch.float32)
        else:
            return torch.full((max_len,), pad_val, dtype=torch.float32)
    elif arr.ndim == 1:
        n = arr.shape[0]
        if n >= max_len:
            return arr[:max_len]
        pad = torch.full((max_len-n,), pad_val, dtype=arr.dtype)
        return torch.cat([arr, pad], dim=0)
    elif arr.ndim == 2:
        n, d = arr.shape
        if n >= max_len:
            return arr[:max_len]
        pad = torch.full((max_len-n, d), pad_val, dtype=arr.dtype)
        return torch.cat([arr, pad], dim=0)
    else:
        raise ValueError(f"Unexpected tensor ndim: {arr.shape}")
    
def pad_to_max_tensor_offset(tensor, max_len, pad_val=0.0):
    """
    把形如 [8] 或 [[8 floats], ...] 的 offset 数据 pad 到 [max_len, 8]
    """
    arr = torch.as_tensor(tensor, dtype=torch.float32)
    if arr.numel() == 0:
        return torch.full((max_len, 8), pad_val, dtype=torch.float32)
    
    if arr.ndim == 1:
        if arr.shape[0] != 8:
            raise ValueError(f"Offset vector must have length 8, got {arr.shape[0]}")
        arr = arr.unsqueeze(0)  # 变成 [1, 8]
        
    if arr.ndim == 2 and arr.shape[1] != 8:
        raise ValueError(f"Expected offsets shape (*, 8), got {arr.shape}")
    
    n = arr.shape[0]
    if n >= max_len:
        return arr[:max_len]
    
    pad = torch.full((max_len - n, 8), pad_val, dtype=torch.float32)
    return torch.cat([arr, pad], dim=0)

def pad_to_max_tensor_box(tensor, max_len, pad_val: float = 0.0, target_dim: int = 4, dtype=torch.float32):
    """
    用于panel/dialog/character bbox等，pad到[max_len, target_dim]，始终返回float32。
    """
    arr = torch.as_tensor(tensor, dtype=dtype)
    if arr.numel() == 0:
        return torch.full((max_len, target_dim), pad_val, dtype=dtype)

    # 一维情况
    if arr.ndim == 1:
        if arr.shape[0] == target_dim:
            arr = arr.unsqueeze(0)  # 单个框
        else:
            # 长度不对，补齐到target_dim
            diff = target_dim - arr.shape[0]
            if diff > 0:
                arr = torch.cat([arr, torch.full((diff,), pad_val, dtype=dtype)])
            arr = arr.unsqueeze(0)

    # 二维但列数不对
    if arr.ndim == 2 and arr.shape[1] != target_dim:
        raise ValueError(f"pad_to_max_tensor_box: expected dim[1]={target_dim}, got {arr.shape[1]}")

    n = arr.shape[0]
    if n >= max_len:
        return arr[:max_len]
    pad = torch.full((max_len - n, target_dim), pad_val, dtype=dtype)
    return torch.cat([arr, pad], dim=0)

def pad_to_max_tensor_scalar(tensor: list, max_len: int, pad_val: float = 0.0):
    arr = torch.as_tensor(tensor)
    if arr.numel() == 0:
        return torch.full((max_len,), pad_val, dtype=torch.float32)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(-1)
    n = arr.shape[0]
    if n >= max_len:
        return arr[:max_len]
    pad = torch.full((max_len-n,), pad_val, dtype=arr.dtype)
    return torch.cat([arr, pad], dim=0)

def pad_to_max_tensor_int(tensor: list, max_len: int, pad_val: int = -1):
    arr = torch.as_tensor(tensor)
    if arr.numel() == 0:
        return torch.full((max_len,), pad_val, dtype=torch.long)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.squeeze(-1)
    n = arr.shape[0]
    if n >= max_len:
        return arr[:max_len]
    pad = torch.full((max_len-n,), pad_val, dtype=arr.dtype)
    return torch.cat([arr, pad], dim=0)



class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.buckets = dataset.buckets
        self.bucket_size_index = dataset.bucket_size_index
        self.batch_size = batch_size
        self.bucket_keys = list(self.buckets.keys())
        self.bucket_batches = self.calculate_bucket_batches()

        self.bucket_samplers = [RandomSampler(self.buckets[bucket_key]) for bucket_key in self.bucket_keys]
        # self.bucket_samplers = [SequentialSampler(self.buckets[bucket_key]) for bucket_key in self.bucket_keys]
        # self.bucket_sampler_iters = [iter(sampler) for sampler in self.bucket_samplers]

    def calculate_bucket_batches(self):
        bucket_batches = []
        for bucket_key in self.bucket_keys:
            batch_size = max(1, round(self.batch_size / (2 ** (self.bucket_size_index[bucket_key] * 2))))
            bucket_length = len(self.buckets[bucket_key])
            bucket_batches.append((bucket_length + batch_size - 1) // batch_size)

        # print(f"rank {accelerator.local_process_index}, bucket_batches: {bucket_batches}")
        return bucket_batches
    
    def get_pseudo_full_batch(self, batch):
        return batch + [None] * (self.batch_size - len(batch))

    def __iter__(self):
        bucket_sampler_iters = [iter(sampler) for sampler in self.bucket_samplers]
        
        batch_bucket_indexes = []
        for idx, num_batch in enumerate(self.bucket_batches):
            batch_bucket_indexes += [idx] * num_batch

        random.shuffle(batch_bucket_indexes)

        for bucket_idx in batch_bucket_indexes:
            bucket_key = self.bucket_keys[bucket_idx]
            batch_size = max(1, round(self.batch_size / (2 ** (self.bucket_size_index[bucket_key] * 2))))
            batch = []
            while True:
                try:
                    idx = next(bucket_sampler_iters[bucket_idx])
                    idx = [bucket_idx, idx]
                    batch.append(idx)
                    if len(batch) == batch_size:
                        # Accelerate seems cannot handle batchsampler with varying batch_sizes in multigpu training.
                        # Pad to the largest batch_size.
                        # print(f"rank {accelerator.local_process_index} yield batch, bucket_key: {bucket_key} batch: {batch} batchsize: {batch_size}")
                        yield self.get_pseudo_full_batch(batch)
                        break
                except StopIteration:
                    # print(f"rank {accelerator.local_process_index} StopIteration, bucket_key: {bucket_key} batch: {batch}")
                    if len(batch) > 0:
                        yield self.get_pseudo_full_batch(batch)
                    break

    def __len__(self):
        return sum(self.bucket_batches)
    
class MangaLayoutDataset(Dataset):
    """
    每条样本=一页（page），frames为panel列表
    采集ip_images为page级(所有panel所有角色融合到一组)
    """
    def __init__(self, ann_source, size_buckets, image_dir=None, tokenizer=None, tokenizer_2=None,
                 max_panels=16, max_num_ips=4, max_num_ip_sources=1,
                 min_ip_height=5, min_ip_width=5, ip_flip_rate=0.5):
        self.samples = []
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
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
        # 图像处理器，CLIP规范
        self.image_dir = image_dir
        self.max_panels = max_panels
        self.max_num_ips = max_num_ips
        self.max_num_ip_sources = max_num_ip_sources
        self.min_ip_height = min_ip_height
        self.min_ip_width = min_ip_width
        self.ip_flip_rate = ip_flip_rate
        self.clip_image_processor = CLIPImageProcessor()
        self.magi_image_processor = ViTImageProcessor()
        # build buckets
        self.size_buckets = size_buckets
        self.buckets = {}
        self.bucket_keys = []
        self.bucket_size_index = {}  # ★ 新增
        self.partition_data()

    def partition_data(self):
        self.buckets = defaultdict(list)
        self.bucket_size_index = {}
        for idx, ann in enumerate(self.samples):
            w, h = ann["width"], ann["height"]
            bh, bw, size_idx = get_bucket_size(h, w, self.size_buckets)
            key = (bh, bw)
            self.buckets[key].append(idx)
            self.bucket_size_index[key] = size_idx  # ★ 必须有
        self.bucket_keys = list(self.buckets.keys())

    def __len__(self):
        # ★ 返回所有桶的总样本量
        return sum(len(v) for v in self.buckets.values())

    def _get_page_image(self, ann):
        if self.image_dir is None: return None
        image_path = os.path.join(self.image_dir, ann["image_path"])
        try:
            page_img = Image.open(image_path).convert("RGB")
        except:
            page_img = Image.new("RGB", (224,224), (0,0,0))
        return page_img

    def _sample_ip_images_page(self, ann):
        """
        page级采集所有panel的所有角色，融合一组，每角色最多max_num_ip_sources张
        输出: ip_images (max_num_ips*max_num_ip_sources, 3, 224, 224) 和配套mask
        返回 PIL.Image 列表，不做 tensor 转换
        """
        characters = []
        for frame in ann.get('frames', []):
            characters.extend(frame.get("characters", []))
        # 获取所有角色ID，最多max_num_ips
        char_id_set = list({c["id"] for c in characters if "bbox" in c})[:self.max_num_ips]
        page_img = self._get_page_image(ann)
        ip_images = []
        ip_exists = []
        ip_char_ids = []
        for char_id in char_id_set:
            boxes = [c["bbox"] for c in characters if c["id"] == char_id]
            # 只用有效框
            valid_boxes = [b for b in boxes if (b[3]-b[1])>=self.min_ip_height and (b[2]-b[0])>=self.min_ip_width]
            n = 0
            for b in valid_boxes[:self.max_num_ip_sources]:
                x1,y1,x2,y2 = b
                crop = page_img.crop([x1,y1,x2,y2]) if page_img else Image.new("RGB", (224,224), (0,0,0))
                if np.random.rand() < self.ip_flip_rate:
                    crop = ImageOps.mirror(crop)
                ip_images.append(crop)
                ip_char_ids.append(int(char_id))   # 保存这一步采样的角色id
                ip_exists.append(1)
                n += 1
            # 填充
            while n < self.max_num_ip_sources:
                ip_images.append(Image.new("RGB", (224,224), (0,0,0)))
                ip_exists.append(0)
                ip_char_ids.append(0)  # 对应填空, -1/0都可
                n += 1
        # 全部pad到max_num_ips*max_num_ip_sources
        total = self.max_num_ips * self.max_num_ip_sources
        while len(ip_images) < total:
            ip_images.append(Image.new("RGB", (224,224), (0,0,0)))
            ip_exists.append(0)
            ip_char_ids.append(0)
        
        clip_ip_images = self.clip_image_processor(images=ip_images, return_tensors="pt").pixel_values
        magi_ip_images = self.magi_image_processor(images=ip_images, return_tensors="pt").pixel_values
        ip_exists_tensor = torch.tensor(ip_exists, dtype=torch.float32)
        ip_char_ids_tensor = torch.tensor(ip_char_ids, dtype=torch.long)
        return clip_ip_images, magi_ip_images, ip_exists_tensor, ip_char_ids_tensor
    
        # return ip_images, ip_exists, ip_char_ids
    

    def __getitem__(self, idx):
        """ index_tuple = (bucket_idx, sample_in_bucket_idx) """
        bucket_idx, local_idx = idx
        bucket_key = self.bucket_keys[bucket_idx]
        ann_idx = self.buckets[bucket_key][local_idx]
        ann = self.samples[ann_idx]
        page_path = os.path.join(self.image_dir, ann["image_path"])
        page_img = Image.open(page_path).convert("RGB")
        
        # 原图
        page_img = self._get_page_image(ann)

        # ★ 按桶尺寸 resize
        bh, bw = bucket_key
        page_img_resized, crop_tl = resize_and_center_crop(page_img, (bh, bw))

        # 归一化到模型需要的范围，并转成 tensor
        page_tensor = self.clip_image_processor(images=page_img_resized, return_tensors="pt").pixel_values.squeeze(0)


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
            out_panels.append(panel)
        # 在page级采集ip_images/ip_exists
        ip_images, magi_ip_images, ip_exists, ip_char_ids = self._sample_ip_images_page(ann)
        
        return {
            "width": width,
            "height": height,
            "style_vector": style_vec,
            "frames": out_panels,
            # "text_input_ids": text_input_ids,
            # "text_input_ids_2": text_input_ids_2,
                    # ★ 新增 page 图数据
            "page_image": page_tensor,  # 按 bucket 尺寸 resize + tensor
            "page_bucket_size": torch.tensor([bh, bw], dtype=torch.int32),
            "page_crop_tl": torch.tensor(crop_tl, dtype=torch.int32),
            "ip_images": ip_images,   # (max_num_ips*max_num_ip_sources, 3, 224, 224)
            "magi_ip_images": magi_ip_images,
            "ip_exists": ip_exists,
            "ip_char_ids": ip_char_ids
        }

def collate_fn(batch: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    # 取各种长度上限
    max_elem_len      = config.dataset.max_elements
    max_panels        = config.dataset.max_panels
    max_dialogs       = config.dataset.max_dialogs
    max_chars         = config.dataset.max_characters
    max_num_ips       = config.model.vision.num_ips
    max_num_ip_sources= config.model.vision.num_ip_sources
    TYPE_PAD          = config.layout_types.TYPE_PAD
    TYPE_PAGE         = config.layout_types.TYPE_PAGE
    TYPE_PANEL        = config.layout_types.TYPE_PANEL
    TYPE_CHAR         = config.layout_types.TYPE_CHAR
    TYPE_DIALOG       = config.layout_types.TYPE_DIALOG
    shape_map         = {k: v.id for k, v in config.panel_shapes.items()}
    shape_map_dialog  = {k: v.id for k, v in config.bubble_shapes.items()}

    bs = len(batch)
    style_vecs     = []
    element_types  = []
    element_indices= []
    element_local_indices = []
    parent_panel_idx=[]
    panel_captions = []
    panel_bboxes   = []
    panel_offsets  = []
    panel_classes  = []
    dialog_bboxes  = []
    dialog_shapes  = []
    dialog_breakout_labels = []
    dialog_breakout_ratios = []
    dialog_speaker_ids = []
    character_bboxes    = []
    character_ids = []
    character_breakout_labels = []
    character_breakout_ratios = []
    width = []
    height = []
    ip_images_list = []
    magi_ip_images_list = []
    ip_exists = []
    ip_char_ids = []    

              
    for ann in batch:
        W, H = ann["width"], ann["height"]
        width.append(W)
        height.append(H)
        style_vecs.append(ann['style_vector'])
        frames = ann.get("frames", [])
        ip_images_list.append(ann["ip_images"])
        magi_ip_images_list.append(ann["magi_ip_images"])
        ip_exists.append(ann["ip_exists"])
        ip_char_ids.append(ann["ip_char_ids"])
        
        # # 新增ip_char_ids: 保证长度为 num_ips*num_ip_sources（需和上面pad一致，否则np.pad左对齐/追加-1或0至指定shape）
        # id_arr = ann["ip_char_ids"]
        # # 保证长度一致pad
        # if len(id_arr) < config.model.vision.num_ips * config.model.vision.num_ip_sources:
        #     padded = torch.cat([
        #         id_arr,
        #         torch.full((config.model.vision.num_ips * config.model.vision.num_ip_sources - len(id_arr),), 0, dtype=torch.long)
        #     ])
        #     ip_char_ids.append(padded)
        # else:
        #     ip_char_ids.append(id_arr[:config.model.vision.num_ips * config.model.vision.num_ip_sources])
            
        # token序列展平（同前，只panel属性/对话/角色展开）
        # et, ei, parent_idx = [TYPE_PAGE],[0],[-1]
        
        # === token 初始化 - Page token ===
        et, ei, parent_idx, elocal_idx = [TYPE_PAGE],[0],[-1], [-1] # 新增 elocal_idx

        panels, dialogs, chars = [], [], []
        pcaps, pbxs, poffs, pcls = [], [], [], []
        
        
        for pi, p in enumerate(frames):
            # Panel token    
            panels.append({"panel_idx": pi, "frame": p})
            et.append(TYPE_PANEL); ei.append(pi); parent_idx.append(-1); elocal_idx.append(-1)  # Panel占位的local_idx
            # Panel属性  
            bbox = p["bbox"]
            class_bbox = p.get("classification_points")
            four = p.get("four_points")
            if four is None:        # 回退逻辑
                x1, y1, x2, y2 = bbox
                four = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            xyxy_n = _norm_xyxy(bbox, W, H)
            cxcywh_n = _xyxy_to_cxcywh(xyxy_n)
            pbxs.append(cxcywh_n)
            # poffs.append(_offsets_from_four_points(class_bbox, four, W, H))
            # 以 four_points 作为基准，计算 classification_points 相对于 four_points 的偏移量。偏移量已经提前计算，但也可以不提前计算
            # poff = p.get("offsets") if p.get("offsets") else _offsets_from_four_points(class_bbox, four, W, H)
            poff = p["offsets"] if "offsets" in p else [0.0] * 8
            poff_norm = [off / float(max(W, H)) for off in poff]  # 归一化！！！！！！！！！！！！！！！
            poffs.append(poff_norm)
            pcls.append(shape_map.get(p.get("shape_type", "panel_rect"), 0))
            pcaps.append(p.get("caption", ""))
            
            # 为每个Panel内的dialog和character维护一个局部计数器
            local_dialog_idx = 0
            for d in p.get("dialogs", []):
                dialogs.append({"panel_idx": pi, "dialog": d, "local_idx": local_dialog_idx})
                local_dialog_idx += 1
            local_char_idx = 0
            for c in p.get("characters", []):
                chars.append({"panel_idx": pi, "char": c, "local_idx": local_char_idx})
                local_char_idx += 1

        # dialog/char
        dbxs, dlabels, dratios, dshapes = [], [], [], []
        cbxs, clabels, cratios, char_ids = [], [], [], []
        dialog_speakers = []
        
        for j, d in enumerate(dialogs):
            # et.append(TYPE_DIALOG); ei.append(j); parent_idx.append(d["panel_idx"])
            et.append(TYPE_DIALOG); ei.append(j); parent_idx.append(d["panel_idx"]); elocal_idx.append(d["local_idx"])
            dg = d["dialog"]
            xyxy = dg.get("rect_box") or [0, 0, 1, 1]
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
            # 取 speaker_id，如果缺失设为 -1
            speaker_id = dg.get("speaker_id", None)
            dialog_speakers.append(-1 if speaker_id is None else int(speaker_id))
            
        for k, c in enumerate(chars):
            # et.append(TYPE_CHAR); ei.append(k); parent_idx.append(c["panel_idx"])
            et.append(TYPE_CHAR); ei.append(k); parent_idx.append(c["panel_idx"]); elocal_idx.append(c["local_idx"])
            ch = c["char"]
            xyxy = ch.get("bbox") or [0,0,1,1]
            char_id = int(ch.get("id", -1))
            x1,y1,x2,y2 = xyxy
            cx = (x1+x2)/2 / W
            cy = (y1+y2)/2 / H
            w  = (x2-x1) / W
            h  = (y2-y1) / H
            cbxs.append([cx,cy,w,h])
            char_ids.append(char_id)
            br = float(ch.get("breakout_ratio", 0.0))
            cratios.append(br)
            clabels.append(1 if br > 1e-6 else 0)
        
        # === 收集到batch    
        element_types.append(torch.tensor(et, dtype=torch.long))
        element_indices.append(torch.tensor(ei, dtype=torch.long))
        element_local_indices.append(torch.tensor(elocal_idx, dtype=torch.long)) # 收集
        parent_panel_idx.append(torch.tensor(parent_idx, dtype=torch.long))
        
        panel_captions.append(pcaps + [""]*(max_panels-len(pcaps)) if len(pcaps)<max_panels else pcaps[:max_panels])
        panel_bboxes.append(torch.tensor(pbxs, dtype=torch.float32))
        panel_offsets.append(torch.tensor(poffs, dtype=torch.float32))
        panel_classes.append(torch.tensor(pcls, dtype=torch.long))
        
        dialog_bboxes.append(pad_to_max_tensor_box(dbxs, max_dialogs, pad_val=0))
        dialog_shapes.append(pad_to_max_tensor_int(dshapes, max_dialogs, pad_val=0))
        dialog_breakout_labels.append(pad_to_max_tensor_int(dlabels, max_dialogs, pad_val=0))
        dialog_breakout_ratios.append(pad_to_max_tensor_scalar(dratios, max_dialogs, pad_val=0.0))
        dialog_speaker_ids.append(pad_to_max_tensor_int(dialog_speakers, max_dialogs, pad_val=-1))

        character_bboxes.append(pad_to_max_tensor_box(cbxs, max_chars, pad_val=0))
        character_ids.append(pad_to_max_tensor_int(char_ids, max_chars, pad_val=-1))
        character_breakout_labels.append(pad_to_max_tensor_int(clabels, max_chars, pad_val=0))
        character_breakout_ratios.append(pad_to_max_tensor_scalar(cratios, max_chars, pad_val=0.0))
        
        # text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
        # text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)

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
    element_local_indices = torch.stack([
        pad_to_max_tensor(elocal_idx, max_elem_len, pad_val=0) for elocal_idx in element_local_indices
    ])
    
    # masks
    token_panel_mask     = (element_types == TYPE_PANEL)
    token_dialog_mask    = (element_types == TYPE_DIALOG)
    token_character_mask = (element_types == TYPE_CHAR)
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
    
    panel_bboxes = torch.stack([pad_to_max_tensor_box(pb, max_panels) for pb in panel_bboxes])
    panel_offsets= torch.stack([pad_to_max_tensor_offset(po, max_panels) for po in panel_offsets])
    panel_classes= torch.stack([pad_to_max_tensor_int(pc, max_panels, pad_val=-1) for pc in panel_classes])
    dialog_bboxes = torch.stack(dialog_bboxes)            # [B, max_dialogs, 4]
    dialog_shapes = torch.stack(dialog_shapes).long()      # [B, max_dialogs]
    dialog_breakout_labels = torch.stack(dialog_breakout_labels).long()
    dialog_breakout_ratios = torch.stack(dialog_breakout_ratios)
    dialog_speaker_ids = torch.stack(dialog_speaker_ids).long()  # [B, max_dialogs]
    character_bboxes = torch.stack(character_bboxes)      # [B, max_chars, 4]
    character_ids = torch.stack(character_ids).long()      # [B, max_chars]
    character_breakout_labels = torch.stack(character_breakout_labels).long()
    character_breakout_ratios = torch.stack(character_breakout_ratios)
    
    # 填充到全长S序列的character_ids/dialog_speaker_ids
    B, S = element_types.shape
    # character_ids_fullseq
    character_ids_fullseq = torch.full((B, S), -1, dtype=torch.long)
    for b in range(B):
        char_positions = torch.nonzero(element_types[b] == TYPE_CHAR, as_tuple=False).squeeze(-1)
        num_fill = min(len(char_positions), character_ids[b].shape[0])
        if num_fill > 0:
            character_ids_fullseq[b, char_positions[:num_fill]] = character_ids[b][:num_fill].long()
    character_ids = character_ids_fullseq

    # dialog_speaker_ids_fullseq
    dialog_speaker_ids_fullseq = torch.full((B, S), -1, dtype=torch.long)
    for b in range(B):
        dialog_positions = torch.nonzero(element_types[b] == TYPE_DIALOG, as_tuple=False).squeeze(-1)
        num_fill = min(len(dialog_positions), dialog_speaker_ids[b].shape[0])
        if num_fill > 0:
            dialog_speaker_ids_fullseq[b, dialog_positions[:num_fill]] = dialog_speaker_ids[b][:num_fill].long()
    character_ids = character_ids_fullseq
    dialog_speaker_ids = dialog_speaker_ids_fullseq
    
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
    height = torch.tensor(height, dtype=torch.int64)
    aspect_ratios = torch.tensor([float(w)/float(h) for w, h in zip(width, height)], dtype=torch.float32)
    # page级ip图片组装成(B, max_num_ips*max_num_ip_sources, 3, 224, 224)?
    ip_images_tensor = torch.stack(ip_images_list, dim=0)
    magi_ip_images_tensor = torch.stack(magi_ip_images_list, dim=0)  # shape 同 ip_images
    # ip_exists = torch.stack(ip_exists, dim=0)
    ip_exists = torch.stack(ip_exists, dim=0).view(bs, max_num_ips, max_num_ip_sources)  # 三维
    ip_char_ids = torch.stack(ip_char_ids, dim=0).view(bs, max_num_ips * max_num_ip_sources)   # 不分source时可(B, max_num_ips)

    batch_dict = {
        "width": width,
        "height": height,
        "aspect_ratios": aspect_ratios,
        "style_vector": style_vector,
        "element_types": element_types,
        "element_indices": element_indices,
        "element_local_indices": element_local_indices,
        "parent_panel_indices": parent_panel_indices,
        "panel_captions": panel_captions,
        "panel_bboxes": panel_bboxes,
        "panel_offsets": panel_offsets,
        "panel_classes": panel_classes,
        "ip_images": ip_images_tensor,
        "magi_ip_images": magi_ip_images_tensor,
        "ip_exists": ip_exists,
        "ip_char_ids": ip_char_ids,
        "dialog_bboxes": dialog_bboxes,
        "dialog_shapes": dialog_shapes,
        "dialog_breakout_labels": dialog_breakout_labels,
        "dialog_breakout_ratios": dialog_breakout_ratios,
        "dialog_speaker_ids": dialog_speaker_ids,
        "character_ids": character_ids,
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
    
    return batch_dict