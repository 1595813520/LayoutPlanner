# infer_real.py
import os
import json
import yaml
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel
from models.layout_planner.planner import LayoutPlanner

# ---- helper utils ----
def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_element_sequence(num_panels, num_dialogs, num_chars, layout_types):
    """
    Build element_types, element_indices, parent_panel_indices like collate.single_collate_fn does.
    We set parent_panel_indices for dialogs/chars to -1 by default.
    """
    TYPE_PAGE = layout_types["TYPE_PAGE"]
    TYPE_PANEL = layout_types["TYPE_PANEL"]
    TYPE_DIALOG = layout_types["TYPE_DIALOG"]
    TYPE_CHAR = layout_types["TYPE_CHAR"]

    element_types = [TYPE_PAGE]
    element_indices = [0]
    parent_panel_indices = [-1]

    # panels
    for i in range(num_panels):
        element_types.append(TYPE_PANEL)
        element_indices.append(i)
        parent_panel_indices.append(-1)

    # dialogs
    for j in range(num_dialogs):
        element_types.append(TYPE_DIALOG)
        element_indices.append(j)
        parent_panel_indices.append(-1)  # unknown parent -> -1

    # chars
    for k in range(num_chars):
        element_types.append(TYPE_CHAR)
        element_indices.append(k)
        parent_panel_indices.append(-1)

    return element_types, element_indices, parent_panel_indices

def cxywh_to_xyxy_pixels(cxywh, W, H):
    # cxywh: tensor or list [cx,cy,w,h] normalized (cx relative to W, cy relative to H, w relative to W, h relative to H)
    cx, cy, w, h = cxywh
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return [float(x1), float(y1), float(x2), float(y2)]

def offsets_to_four_points(base_xyxy, offsets, W, H):
    """
    base_xyxy: [x1,y1,x2,y2] in pixels for this panel
    offsets: length-8 tensor/list [dx1,dy1,dx2,dy2,...] where dx/dy are normalized by scale=max(W,H)
    returns: list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in pixels (float)
    base order is same as collate: [ (x1,y1),(x2,y1),(x2,y2),(x1,y2) ]
    """
    if len(offsets) < 8:
        # fallback: return base rectangle corners
        x1,y1,x2,y2 = base_xyxy
        return [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    scale = float(max(W, H))
    x1,y1,x2,y2 = base_xyxy
    bases = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    pts = []
    for i in range(4):
        bx, by = bases[i]
        dx = float(offsets[2*i]) * scale
        dy = float(offsets[2*i+1]) * scale
        pts.append([bx + dx, by + dy])
    return pts

def tensor_to_list(t):
    if t is None:
        return []
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().tolist()
    return list(t)

# ---- main inference routine ----
def infer_and_save(cfg_path, ckpt_path, test_json, image_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_cfg(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    text_encoder_path = cfg["dataset"]["train"]["text_encoder_path"]
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(text_encoder_path)
    text_encoder.eval()

    # read mappings
    layout_types = cfg.get("layout_types", {"TYPE_PAD":0,"TYPE_PAGE":1,"TYPE_PANEL":2,"TYPE_CHAR":3,"TYPE_DIALOG":4})
    panel_shapes_cfg = cfg.get("panel_shapes", {})
    # build id->name map (panel_shapes: name -> {id: x})
    id2panel = {v["id"]: k for k, v in panel_shapes_cfg.items()}

    bubble_shapes_cfg = cfg.get("bubble_shapes", {})
    id2bubble = {v["id"]: k for k, v in bubble_shapes_cfg.items()}

    # build model
    model_params = cfg["model"]["parameters"]
    max_panels = cfg["dataset"]["parameters"]["max_panels"]
    
    planner = LayoutPlanner(
        encoder_cfg={**model_params, "layout_types": layout_types},
        heads_cfg={
            "num_panel_classes": len(panel_shapes_cfg),
            "num_dialog_shapes": len(bubble_shapes_cfg),
        }
    ).to(device)

    ck = torch.load(ckpt_path, map_location=device)
    # ck may be saved as {"model": state_dict, "epoch": ...}
    state = ck.get("model", ck)
    planner.load_state_dict(state)
    planner.eval()

    # load test list
    with open(test_json, "r", encoding="utf-8") as f:
        test_list = json.load(f)

    results = []
    for sample in test_list:
        img_rel = sample["image_path"]
        img_path = os.path.join(image_dir, img_rel) if not os.path.isabs(img_rel) else img_rel
        W = sample.get("width", None)
        H = sample.get("height", None)
        if (W is None) or (H is None):
            # try to load image to get size
            with Image.open(img_path) as tmp:
                W, H = tmp.size

        # build element sequence from requested counts (fallback to 3 panels if not provided)
        num_panels = sample.get("num_panels", len(sample.get("frames", [])) if "frames" in sample else 3)
        num_dialogs = sample.get("num_dialogs", 0)
        num_chars = sample.get("num_characters", 0)

        etypes, eindices, pidxs = build_element_sequence(num_panels, num_dialogs, num_chars, layout_types)

        # tensors (batch size 1)
        etypes_t = torch.tensor([etypes], dtype=torch.long, device=device)
        eind_t = torch.tensor([eindices], dtype=torch.long, device=device)
        pidxs_t = torch.tensor([pidxs], dtype=torch.long, device=device)

        # style vector (B,4)
        sp = sample.get("style_parameters", {})
        style_vec = torch.tensor([[
            float(sp.get("layout_density", 0.5)),
            float(sp.get("alignment_score", 0.5)),
            float(sp.get("shape_instability", 0.0)),
            float(sp.get("breakout_intensity", 0.0)),
        ]], dtype=torch.float32, device=device)
        
        if "panel_captions" in sample and isinstance(sample["panel_captions"], list):
            panel_captions = sample["panel_captions"]
        elif "frames" in sample:
            panel_captions = [fr.get("caption", "") for fr in sample["frames"]]
        else:
            panel_captions = [""] * num_panels

        # pad 到 max_panels
        if len(panel_captions) < max_panels:
            panel_captions += [""] * (max_panels - len(panel_captions))
        elif len(panel_captions) > max_panels:
            panel_captions = panel_captions[:max_panels]

        # text encode (B=1, max_panels, D)
        inputs = tokenizer(panel_captions, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            txt_emb = text_encoder(**inputs).last_hidden_state[:, 0, :]   # (max_panels, D)
        panel_caption_embeddings = txt_emb.unsqueeze(0)  # (1, max_panels, D)

        batch = {
            "element_types": etypes_t,
            "element_indices": eind_t,
            "parent_panel_indices": pidxs_t,
            "style_vector": style_vec,
            "panel_caption_embeddings": panel_caption_embeddings
        }

        
        # forward
        with torch.no_grad():
            outputs = planner(batch)  # list of per-sample outputs
            
        p_out = {
            "bbox": outputs["panel_bbox"][0],  # 新增：取第一个批次的数据
            "offsets": outputs["panel_offsets"][0],  # 新增：取第一个批次的数据
            "class_logits": outputs["panel_class_logits"][0]  # 新增：取第一个批次的数据
        }
        
        # 对话框数据：取第一个批次的所有对话框
        d_out = {
            "bbox": outputs["dialog_bbox"][0],  # 新增：取第一个批次的数据
            "breakout_ratio": outputs["dialog_breakout_ratio"][0],
            "shape_logits": outputs["dialog_shape_logits"][0]
        }
        
        # 角色数据：取第一个批次的所有角色
        c_out = {
            "bbox": outputs["character_bbox"][0],  # 新增：取第一个批次的数据
            "breakout_ratio": outputs["character_breakout_ratio"][0]
        }

        # p_out, d_out, c_out = outputs[0]

        # panels
        panel_bboxes = tensor_to_list(p_out.get("bbox"))          # (P,4) normalized cxywh
        panel_offsets = tensor_to_list(p_out.get("offsets"))     # (P,8) raw
        panel_class_logits = tensor_to_list(p_out.get("class_logits"))  # (P, num_classes)

        panels = []
        P = len(panel_bboxes)
        for i in range(P):
            cxywh = panel_bboxes[i]  # [cx,cy,w,h] normalized
            base_xyxy = cxywh_to_xyxy_pixels(cxywh, W, H)
            offs = panel_offsets[i] if i < len(panel_offsets) else [0.0]*8
            four_pts = offsets_to_four_points(base_xyxy, offs, W, H)
            cls_id = int(max(range(len(panel_class_logits[i])), key=lambda k: panel_class_logits[i][k])) if panel_class_logits and len(panel_class_logits)>i else None
            panels.append({
                "bbox_xyxy": base_xyxy,                # pixel x1,y1,x2,y2 (from bbox)
                "cxywh_norm": panel_bboxes[i],
                "offsets": offs,
                "four_points": four_pts,
                "panel_class_id": cls_id,
                "panel_class_name": id2panel.get(cls_id, None)
            })

        # dialogs
        dialogs = []
        if d_out is not None:
            dbbox = tensor_to_list(d_out.get("bbox"))  # (D,4) cxywh normalized
            dbreak = tensor_to_list(d_out.get("breakout_ratio"))  # (D,) or (D,1)
            dshape_logits = tensor_to_list(d_out.get("shape_logits"))  # (D, n_shapes)
            D = len(dbbox)
            for i in range(D):
                cxywh = dbbox[i]
                xyxy = cxywh_to_xyxy_pixels(cxywh, W, H)
                # 如果是 list of list 或 tensor (D,1)
                val = dbreak[i]
                if isinstance(val, (list, tuple)):
                    val = val[0]  # 取第0个元素
                br = float(val)

                shape_id = int(max(range(len(dshape_logits[i])), key=lambda k: dshape_logits[i][k])) if dshape_logits and len(dshape_logits)>i else None
                dialogs.append({
                    "bbox_xyxy": xyxy,
                    "cxywh_norm": cxywh,
                    "breakout_ratio": br,
                    "shape_id": shape_id,
                    "shape_name": id2bubble.get(shape_id, None)
                })

        # characters
        characters = []
        if c_out is not None:
            cbbox = tensor_to_list(c_out.get("bbox"))
            cbr = tensor_to_list(c_out.get("breakout_ratio"))
            C = len(cbbox)
            for i in range(C):
                cxywh = cbbox[i]
                xyxy = cxywh_to_xyxy_pixels(cxywh, W, H)
                val = cbr[i]
                if isinstance(val, (list, tuple)):
                    val = val[0]
                br = float(val)

                characters.append({
                    "bbox_xyxy": xyxy,
                    "cxywh_norm": cxywh,
                    "breakout_ratio": br
                })

        result = {
            "image_path": img_rel,
            "width": W, "height": H,
            "panels": panels,
            "dialogs": dialogs,
            "characters": characters
        }
        results.append(result)

        # visualization
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        # panel polygons and bbox
        for p in panels:
            # polygon
            poly = [tuple(map(float, pt)) for pt in p["four_points"]]
            draw.polygon(poly, outline=(255,0,0))
            # bbox rect
            x1,y1,x2,y2 = p["bbox_xyxy"]
            draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
            if p["panel_class_name"]:
                draw.text((x1+2, y1+2), str(p["panel_class_name"]), fill=(255,0,0))
        # dialogs blue, chars green
        for d in dialogs:
            x1,y1,x2,y2 = d["bbox_xyxy"]
            draw.rectangle([x1,y1,x2,y2], outline=(0,0,255), width=2)
        for c in characters:
            x1,y1,x2,y2 = c["bbox_xyxy"]
            draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)

        out_vis = os.path.join(out_dir, f"vis_1{Path(img_rel).stem}.png")
        img.save(out_vis)

    # save results json
    with open(os.path.join(out_dir, "inference_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done. Results saved to:", out_dir)


# ---- run as script ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/planner.yaml")
    parser.add_argument("--checkpoint", required=True, help="path to planner_best.pt or planner_epochX.pt")
    parser.add_argument("--test_json", required=True, help="test input list json")
    parser.add_argument("--image_dir", default=".", help="root dir for image_path entries")
    parser.add_argument("--output_dir", default="infer_outputs")
    args = parser.parse_args()
    infer_and_save(args.config, args.checkpoint, args.test_json, args.image_dir, args.output_dir)
