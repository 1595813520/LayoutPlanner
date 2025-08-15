# utils/collate.py
import torch
from typing import Dict, Any, List

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

def single_collate_fn(ann: Dict[str, Any], cfg: Dict[str, int]) -> Dict[str, Any]:
    W, H = ann["width"], ann["height"]
    frames = ann.get("frames", [])
    style = ann["style_parameters"]

    TYPE_PAD    = cfg["layout_types"]["TYPE_PAD"]
    TYPE_PAGE   = cfg["layout_types"]["TYPE_PAGE"]
    TYPE_PANEL  = cfg["layout_types"]["TYPE_PANEL"]
    TYPE_CHAR   = cfg["layout_types"]["TYPE_CHAR"]
    TYPE_DIALOG = cfg["layout_types"]["TYPE_DIALOG"]

    element_types = [TYPE_PAGE]
    element_indices = [0]
    parent_panel_idx = [-1]
    panels, dialogs, chars = [], [], []

    for pi, p in enumerate(frames):
        panels.append({"panel_idx": pi, "frame": p})
        for d in p.get("dialogs", []):
            dialogs.append({"panel_idx": pi, "dialog": d})
        for c in p.get("characters", []):
            chars.append({"panel_idx": pi, "char": c})

    for i, _ in enumerate(panels):
        element_types.append(TYPE_PANEL); element_indices.append(i); parent_panel_idx.append(-1)
    for j, d in enumerate(dialogs):
        element_types.append(TYPE_DIALOG); element_indices.append(j); parent_panel_idx.append(d["panel_idx"])
    for k, c in enumerate(chars):
        element_types.append(TYPE_CHAR); element_indices.append(k); parent_panel_idx.append(c["panel_idx"])

    # Panels -> 统一到 (cx, cy, w, h)，全部归一化到 [0,1]
    panel_bboxes, panel_offsets, panel_classes = [], [], []
    shape_map = {k: v["id"] for k, v in cfg["panel_shapes"].items()}
    for p in panels:
        fr = p["frame"]
        bbox = fr["bbox"]
        four = fr.get("four_points")
        if four is None:
            x1, y1, x2, y2 = bbox
            four = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

        # 先 xyxy 归一化，再转 cxcywh
        xyxy_n = _norm_xyxy(bbox, W, H)
        cxcywh_n = _xyxy_to_cxcywh(xyxy_n)
        panel_bboxes.append(cxcywh_n)

        panel_offsets.append(_offsets_from_four_points(four, bbox, W, H))
        panel_classes.append(shape_map.get(fr.get("shape_type", "panel_rect"), 0))

    # Dialogs
    dialog_bboxes, dialog_break_labels, dialog_break_ratios, dialog_shapes = [], [], [], []
    shape_map_dialog = {"bubble_oval":0, "bubble_flower":1, "bubble_burst":2, "bubble_rect":3}
    for d in dialogs:
        dg = d["dialog"]
        xyxy = dg.get("dialog_bbox") or dg.get("bbox") or [0, 0, 1, 1]
        x1,y1,x2,y2 = xyxy
        cx = (x1+x2)/2 / W
        cy = (y1+y2)/2 / H
        w  = (x2-x1) / W
        h  = (y2-y1) / H
        dialog_bboxes.append([cx,cy,w,h])
        br = float(dg.get("breakout_ratio", 0.0))
        dialog_break_ratios.append(br)
        dialog_break_labels.append(1 if br > 1e-6 else 0)
        dialog_shapes.append(shape_map_dialog.get(dg.get("bubble_type", None), 0))

    # Characters
    char_bboxes, char_break_labels, char_break_ratios = [], [], []
    for c in chars:
        ch = c["char"]
        xyxy = ch.get("bbox") or [0,0,1,1]
        x1,y1,x2,y2 = xyxy
        cx = (x1+x2)/2 / W
        cy = (y1+y2)/2 / H
        w  = (x2-x1) / W
        h  = (y2-y1) / H
        char_bboxes.append([cx,cy,w,h])
        br = float(ch.get("breakout_ratio", 0.0))
        char_break_ratios.append(br)
        char_break_labels.append(1 if br > 1e-6 else 0)

    style_vec = torch.tensor([
        style["layout_density"],
        style["alignment_score"],
        style["shape_instability"],
        style["breakout_intensity"],
    ], dtype=torch.float32)

    return {
        "width": W,
        "height": H,
        "style_vector": style_vec,
        "element_types": torch.tensor(element_types, dtype=torch.long),
        "element_indices": torch.tensor(element_indices, dtype=torch.long),
        "parent_panel_indices": torch.tensor(parent_panel_idx, dtype=torch.long),
        "targets": {
            "panel_bboxes": torch.tensor(panel_bboxes, dtype=torch.float32) if panel_bboxes else torch.empty((0,4)),
            "panel_offsets": torch.tensor(panel_offsets, dtype=torch.float32) if panel_offsets else torch.empty((0,8)),
            "panel_classes": torch.tensor(panel_classes, dtype=torch.long) if panel_classes else torch.empty((0,), dtype=torch.long),
            "dialog_bboxes": torch.tensor(dialog_bboxes, dtype=torch.float32) if dialog_bboxes else torch.empty((0,4)),
            "dialog_shapes": torch.tensor(dialog_shapes, dtype=torch.long) if dialog_shapes else torch.empty((0,), dtype=torch.long),
            "dialog_breakout_labels": torch.tensor(dialog_break_labels, dtype=torch.long) if dialog_break_labels else torch.empty((0,), dtype=torch.long),
            "dialog_breakout_ratios": torch.tensor(dialog_break_ratios, dtype=torch.float32) if dialog_break_ratios else torch.empty((0,)),
            "character_bboxes": torch.tensor(char_bboxes, dtype=torch.float32) if char_bboxes else torch.empty((0,4)),
            "character_breakout_labels": torch.tensor(char_break_labels, dtype=torch.long) if char_break_labels else torch.empty((0,), dtype=torch.long),
            "character_breakout_ratios": torch.tensor(char_break_ratios, dtype=torch.float32) if char_break_ratios else torch.empty((0,)),
        }
    }

def collate_fn(batch: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    proc = [single_collate_fn(b, cfg) for b in batch]

    max_elem_len = max(p["element_types"].shape[0] for p in proc)
    max_panels   = max(p["targets"]["panel_bboxes"].shape[0] for p in proc)
    max_dialogs  = max(p["targets"]["dialog_bboxes"].shape[0] for p in proc)
    max_chars    = max(p["targets"]["character_bboxes"].shape[0] for p in proc)

    out = {
        "width": torch.tensor([p["width"] for p in proc], dtype=torch.int64),
        "height": torch.tensor([p["height"] for p in proc], dtype=torch.int64),
        "style_vector": torch.stack([p["style_vector"] for p in proc], dim=0),
        "element_types": torch.stack([pad_to_max_tensor(p["element_types"], max_elem_len,
                                                        pad_val=cfg["layout_types"]["TYPE_PAD"]) for p in proc]),
        "element_indices": torch.stack([pad_to_max_tensor(p["element_indices"], max_elem_len, pad_val=0) for p in proc]),
        "parent_panel_indices": torch.stack([pad_to_max_tensor(p["parent_panel_indices"], max_elem_len, pad_val=-1) for p in proc]),
        "targets": {
            "panel_bboxes": torch.stack([pad_to_max_tensor(t["panel_bboxes"], max_panels) for t in [p["targets"] for p in proc]]),
            "panel_offsets": torch.stack([pad_to_max_tensor(t["panel_offsets"], max_panels) for t in [p["targets"] for p in proc]]),
            "panel_classes": torch.stack([pad_to_max_tensor(t["panel_classes"], max_panels, pad_val=-1) for t in [p["targets"] for p in proc]]),
            "dialog_bboxes": torch.stack([pad_to_max_tensor(t["dialog_bboxes"], max_dialogs) for t in [p["targets"] for p in proc]]),
            "dialog_shapes": torch.stack([pad_to_max_tensor(t["dialog_shapes"], max_dialogs, pad_val=-1) for t in [p["targets"] for p in proc]]),
            "dialog_breakout_labels": torch.stack([pad_to_max_tensor(t["dialog_breakout_labels"], max_dialogs, pad_val=0) for t in [p["targets"] for p in proc]]),
            "dialog_breakout_ratios": torch.stack([pad_to_max_tensor(t["dialog_breakout_ratios"], max_dialogs, pad_val=0.0) for t in [p["targets"] for p in proc]]),
            "character_bboxes": torch.stack([pad_to_max_tensor(t["character_bboxes"], max_chars) for t in [p["targets"] for p in proc]]),
            "character_breakout_labels": torch.stack([pad_to_max_tensor(t["character_breakout_labels"], max_chars, pad_val=0) for t in [p["targets"] for p in proc]]),
            "character_breakout_ratios": torch.stack([pad_to_max_tensor(t["character_breakout_ratios"], max_chars, pad_val=0.0) for t in [p["targets"] for p in proc]]),
        }
    }
    return out
