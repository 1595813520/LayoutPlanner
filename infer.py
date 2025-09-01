import os, json, torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from transformers import CLIPImageProcessor, ViTImageProcessor
from src.pipeline_planner import load_models, build_element_sequence
from src.utils import mean_multiple_ip_embeds

def clip_box(xyxy, W, H):
    """将坐标裁剪到画布范围"""
    return [max(0, min(W, xyxy[0])),
            max(0, min(H, xyxy[1])),
            max(0, min(W, xyxy[2])),
            max(0, min(H, xyxy[3]))]

def cxywh_to_xyxy_pixels(cxywh, W, H):
    """从归一化cxywh恢复到像素坐标"""
    cx, cy, w, h = [max(0.0, min(1.0, float(v))) for v in cxywh]  # clamp到[0,1]
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return clip_box([x1, y1, x2, y2], W, H)

def offsets_to_four_points(base_xyxy, offsets, W, H):
    scale = float(max(W, H))
    x1, y1, x2, y2 = base_xyxy
    bases = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    pts = []
    for i, (bx, by) in enumerate(bases):
        dx = float(offsets[2*i]) * scale
        dy = float(offsets[2*i+1]) * scale
        pts.append([bx + dx, by + dy])
    return pts

def visualize_prediction(image_path, frames, save_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for f in frames:
        draw.polygon([tuple(pt) for pt in f["four_points"]], outline="red", width=2)
        for ch in f["characters"]:
            draw.rectangle(ch["bbox"], outline="green", width=2)
        for dg in f["dialogs"]:
            draw.rectangle(dg["bbox"], outline="blue", width=2)
    img.save(save_path)
    print(f"🖼️ Visualization saved to {save_path}")

def assign_to_panel(obj_bbox, panels):
    ox = (obj_bbox[0] + obj_bbox[2]) / 2
    oy = (obj_bbox[1] + obj_bbox[3]) / 2
    for i, (px1, py1, px2, py2) in enumerate(panels):
        if px1 <= ox <= px2 and py1 <= oy <= py2:
            return i
    return 0  # fallback

def run_infer(cfg_path, ckpt_path, test_json, image_dir, output_dir):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, text_encoder, image_encoder, magi_image_encoder, resampler, planner = load_models(cfg, device)
    planner.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])

    clip_proc = CLIPImageProcessor()
    magi_proc = ViTImageProcessor()

    with open(test_json, "r") as f:
        samples = json.load(f)

    results = []
    for smp in samples:
        npan, nch, ndlg = smp["num_panels"], smp["num_characters"], smp["num_dialogs"]

        # 文本特征
        caps = smp.get("panel_captions", [""] * npan)
        if len(caps) < cfg.dataset.max_panels:
            caps += [""] * (cfg.dataset.max_panels - len(caps))
        inputs = tokenizer(caps, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
        txt_emb = text_encoder(**inputs).last_hidden_state[:, 0, :]
        panel_caption_embeddings = txt_emb.unsqueeze(0)

        # 视觉特征
        N_ips = cfg.model.vision.num_ips
        N_src = cfg.model.vision.num_ip_sources
        B = 1
        ip_images, ip_exists_mask, ip_char_ids = [], [], []
        for cid in range(N_ips):
            if cid < len(smp["characters"]):
                img_path = os.path.join(image_dir, smp["characters"][cid]["ip_image_path"])
                img_tensor = clip_proc(Image.open(img_path).convert("RGB"), return_tensors="pt")["pixel_values"].squeeze(0)
                ip_images.append(img_tensor)
                ip_exists_mask.append(1)
                ip_char_ids.append(smp["characters"][cid]["id"])
            else:
                ip_images.append(torch.zeros((3,224,224)))
                ip_exists_mask.append(0)
                ip_char_ids.append(0)

        ip_images_tensor = torch.stack(ip_images, dim=0).view(B, N_ips, 3, 224, 224)
        ip_exists_tensor = torch.tensor(ip_exists_mask, dtype=torch.float32, device=device).view(B, N_ips, 1).expand(-1,-1,N_src)
        ip_char_ids_tensor = torch.tensor(ip_char_ids, dtype=torch.long, device=device)

        # 展平视觉特征
        ip_images_flat = ip_images_tensor.unsqueeze(2).expand(-1, -1, N_src, -1, -1, -1).reshape(B*N_ips*N_src, 3, 224, 224)
        outputs_img = image_encoder(ip_images_flat.to(device))
        image_embeds_raw = outputs_img.last_hidden_state.view(B, N_ips, N_src, outputs_img.last_hidden_state.shape[1], outputs_img.last_hidden_state.shape[2])
        image_embeds_raw = image_embeds_raw.transpose(1,2).contiguous().view(B*N_src, N_ips, outputs_img.last_hidden_state.shape[1], outputs_img.last_hidden_state.shape[2])

        if magi_image_encoder is not None:
            magi_images_flat = ip_images_flat
            magi_hidden = magi_image_encoder(magi_images_flat.to(device)).last_hidden_state
            magi_embeds = magi_hidden[:,0].view(B, N_ips, N_src, -1).transpose(1,2).contiguous().view(B*N_src, N_ips, -1)
        else:
            magi_embeds = None

        image_embeds_all = resampler(image_embeds_raw, magi_embeds)
        image_embeds_final = mean_multiple_ip_embeds(image_embeds_all, ip_exists_tensor, cfg, B)

        # 对齐到max_elements
        max_elements = cfg.dataset.max_elements
        TYPE_CHAR = cfg.layout_types.TYPE_CHAR
        et, ei, pidx = build_element_sequence(npan, ndlg, nch, cfg.layout_types, max_elements)
        char_positions = torch.nonzero(et[0] == TYPE_CHAR, as_tuple=False).squeeze(-1)
        character_ids_full = torch.full((1, max_elements), -1, dtype=torch.long)
        char_vis_full = torch.zeros((1, max_elements, image_embeds_final.shape[-1]))
        num_fill = min(len(char_positions), len(ip_char_ids))
        if num_fill > 0:
            character_ids_full[0, char_positions[:num_fill]] = ip_char_ids_tensor[:num_fill].cpu()
            char_vis_full[0, char_positions[:num_fill]] = image_embeds_final[0, :num_fill].cpu()

        batch = {
            "element_types": et, "element_indices": ei, "parent_panel_indices": pidx,
            "element_local_indices": torch.full_like(et, -1),
            "dialog_speaker_ids": torch.full_like(et, -1),
            "style_vector": torch.tensor([list(smp["style_parameters"].values())], dtype=torch.float32),
            "aspect_ratios": torch.tensor([smp["width"]/smp["height"]], dtype=torch.float32),
            "panel_caption_embeddings": panel_caption_embeddings.cpu(),
            "character_visual_embeddings": char_vis_full,
            "character_ids": character_ids_full
        }
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # 推理
        outputs = planner(batch)
        id2panel = {v.id: k for k, v in cfg.panel_shapes.items()}
        id2bubble = {v.id: k for k, v in cfg.bubble_shapes.items()}

        # Panels
        panel_bboxes_px, frames = [], []
        for p in range(npan):
            bbox_px = cxywh_to_xyxy_pixels(outputs["panel_bbox"][0][p].tolist(), smp["width"], smp["height"])
            panel_bboxes_px.append(bbox_px)
            offs = outputs["panel_offsets"][0][p].tolist()
            four_pts = offsets_to_four_points(bbox_px, offs, smp["width"], smp["height"])
            cls_id = torch.argmax(outputs["panel_class_logits"][0][p]).item()
            frames.append({"bbox": bbox_px, "offsets": offs, "four_points": four_pts,
                           "panel_class_name": id2panel.get(cls_id,"unknown"),
                           "characters": [], "dialogs": []})

        # Characters
        for idx_token, pos in enumerate(char_positions[:nch]):
            cb = cxywh_to_xyxy_pixels(outputs["character_bbox"][0][idx_token].tolist(), smp["width"], smp["height"])
            pid = assign_to_panel(cb, panel_bboxes_px)
            br = outputs["character_breakout_ratio"][0][idx_token].item()
            char_id_val = int(character_ids_full[0, pos].item())
            frames[pid]["characters"].append({"id": char_id_val, "bbox": cb, "breakout_ratio": br})

        # Dialogs
        dlg_positions = torch.nonzero(et[0] == cfg.layout_types.TYPE_DIALOG, as_tuple=False).squeeze(-1)
        for idx_token, pos in enumerate(dlg_positions[:ndlg]):
            db = cxywh_to_xyxy_pixels(outputs["dialog_bbox"][0][idx_token].tolist(), smp["width"], smp["height"])
            pid = assign_to_panel(db, panel_bboxes_px)
            br = outputs["dialog_breakout_ratio"][0][idx_token].item()
            shid = torch.argmax(outputs["dialog_shape_logits"][0][idx_token]).item()
            frames[pid]["dialogs"].append({"bbox": db, "breakout_ratio": br, "shape_name": id2bubble.get(shid,"unknown"), "speaker_id": -1})

        results.append({"image_path": smp["image_path"], "width": smp["width"], "height": smp["height"], "frames": frames})

        # 可视化
        visualize_prediction(os.path.join(image_dir, smp["image_path"]), frames, os.path.join(output_dir, f"vis_{os.path.basename(smp['image_path'])}"))

    json.dump(results, open(os.path.join(output_dir, "inference_results.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--test_json")
    parser.add_argument("--image_dir")
    parser.add_argument("--output_dir", default="infer_out")
    args = parser.parse_args()
    run_infer(args.config, args.checkpoint, args.test_json, args.image_dir, args.output_dir)