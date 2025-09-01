# infer.py
import os
import json
import torch
import argparse
from PIL import Image, ImageDraw
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    AutoModel
)
from models.layout_planner.planner import LayoutPlanner
from models.layout_planner.resampler import Resampler
from src.utils import load_ckpt, mean_multiple_ip_embeds

def load_config(path: str) -> DictConfig:
    """加载 YAML 配置文件"""
    return OmegaConf.load(path)

def build_element_sequence(num_panels: int, num_dialogs: int, num_chars: int, layout_types: DictConfig):
    """构建基础的元素序列 (types, indices, parent_indices)"""
    et, ei, pidx = [layout_types.TYPE_PAGE], [0], [-1]
    for i in range(num_panels):
        et.append(layout_types.TYPE_PANEL); ei.append(i); pidx.append(-1)
    for j in range(num_dialogs):
        et.append(layout_types.TYPE_DIALOG); ei.append(j); pidx.append(-1)
    for k in range(num_chars):
        et.append(layout_types.TYPE_CHAR); ei.append(k); pidx.append(-1)
    return et, ei, pidx

def process_visual_features(
    character_info: list,
    config: DictConfig,
    image_encoder: CLIPVisionModelWithProjection,
    image_processor: CLIPImageProcessor,
    magi_image_encoder: AutoModel,
    resampler: Resampler,
    device: torch.device
):
    """
    加载、处理并编码IP图像，精确复现训练时的视觉特征提取逻辑。
    """
    if not character_info:
        return None, None

    # 从 config 中获取参数
    N_ips = config.model.vision.num_ips
    N_src = config.model.vision.num_ip_sources
    
    ip_images, magi_ip_images, ip_exists, ip_char_ids = [], [], [], []
    
    # 1. 准备输入数据 (ip_images, ip_exists, ip_char_ids)
    valid_chars = sorted([c for c in character_info if c.id != -1], key=lambda x: x.id)
    
    for i in range(N_ips):
        if i < len(valid_chars):
            char = valid_chars[i]
            ip_char_ids.append(char.id)
            # 在推理时，我们假设每个角色只有一张参考图 (N_src=1)
            try:
                img = Image.open(char.ip_image_path).convert("RGB")
                ip_images.append(img)
                if magi_image_encoder is not None:
                    # 假设Magi使用相同的图像，只是处理器不同
                    magi_ip_images.append(img)
                ip_exists.append(1)
            except FileNotFoundError:
                print(f"警告：找不到IP图片 {char.ip_image_path}，将使用空白图片代替。")
                ip_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                if magi_image_encoder is not None:
                    magi_ip_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                ip_exists.append(0)
        else:
            # 用空数据填充至 N_ips
            ip_char_ids.append(-1)
            ip_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            if magi_image_encoder is not None:
                magi_ip_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            ip_exists.append(0)

    # 2. 图像预处理
    ip_images_processed = image_processor(images=ip_images, return_tensors="pt").pixel_values
    if magi_image_encoder is not None:
        magi_images_processed = image_processor(images=magi_ip_images, return_tensors="pt").pixel_values
    
    # 3. 通过编码器和Resampler (精确复现训练逻辑)
    with torch.no_grad():
        B = 1 # 推理时 batch size 为 1
        
        # (B, N_ips * N_src, C, H, W) -> (B * N_ips * N_src, C, H, W)
        ip_images_flat = ip_images_processed.view(B * N_ips * N_src, *ip_images_processed.shape[1:]).to(device)
        
        # CLIP Encoder
        image_embeds_raw = image_encoder(ip_images_flat, output_hidden_states=False, return_dict=True).last_hidden_state
        
        # Reshape for Resampler
        image_embeds_raw = image_embeds_raw.view(B, N_ips, N_src, *image_embeds_raw.shape[1:]).transpose(1, 2).contiguous()
        image_embeds_raw = image_embeds_raw.view(B * N_src, N_ips, *image_embeds_raw.shape[3:])
        
        # MAGI Encoder
        magi_image_embeds = None
        if magi_image_encoder is not None:
            magi_images_flat = magi_images_processed.view(B * N_ips * N_src, *magi_images_processed.shape[1:]).to(device)
            magi_hidden = magi_image_encoder(magi_images_flat).last_hidden_state
            magi_embeds = magi_hidden[:, 0]
            magi_embeds = magi_embeds.view(B, N_ips, N_src, -1).transpose(1, 2)
            magi_image_embeds = magi_embeds.contiguous().view(B * N_src, N_ips, -1)

        # Resampler
        image_embeds_all = resampler(image_embeds_raw, magi_image_embeds)
        
        # Mean across sources
        ip_exists_tensor = torch.tensor(ip_exists, dtype=torch.float32, device=device).view(B, N_ips, N_src)
        image_embeds_final = mean_multiple_ip_embeds(image_embeds_all, ip_exists_tensor, config, B)
        
        # Pool vision tokens
        vis_tokens_per_char = image_embeds_final[:, config.model.vision.num_dummy_tokens:, :]
        character_visual_tokens = vis_tokens_per_char.view(B, N_ips, config.model.vision.num_vision_tokens, -1)
        character_visual_embeddings_sampled = character_visual_tokens.mean(dim=2) # (B, N_ips, D_cross)

    return character_visual_embeddings_sampled, torch.tensor([ip_char_ids], device=device)

def cxywh_to_xyxy_pixels(cxywh, W, H):
    cx, cy, w, h = cxywh
    x1, y1, x2, y2 = (cx - w/2.0) * W, (cy - h/2.0) * H, (cx + w/2.0) * W, (cy + h/2.0) * H
    return [float(x1), float(y1), float(x2), float(y2)]

def offsets_to_four_points(base_xyxy, offsets, W, H):
    scale = float(max(W, H))
    x1, y1, x2, y2 = base_xyxy
    bases = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    pts = [[b[0] + float(offsets[2*i]) * scale, b[1] + float(offsets[2*i+1]) * scale] for i, b in enumerate(bases)]
    return pts

def tensor_to_list(t):
    if t is None: return []
    return t.detach().cpu().tolist() if isinstance(t, torch.Tensor) else list(t)

# =================================================================================
# 主推理函数
# =================================================================================

def infer_and_save(config: DictConfig, ckpt_path: str, test_json_path: str, image_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载所有模型 (与训练脚本的加载逻辑对齐)
    print("正在加载模型...")
    text_encoder = CLIPTextModel.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder").to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer")
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.model.image_encoder_path).to(device).eval()
    # image_processor = CLIPImageProcessor.from_pretrained(config.model.image_encoder_path)
    image_processor = CLIPImageProcessor()

    magi_image_encoder = None
    if config.model.magi_image_encoder_path:
        magi_image_encoder = AutoModel.from_pretrained(config.model.magi_image_encoder_path, trust_remote_code=True).crop_embedding_model.to(device).eval()

    resampler = Resampler(
        dim=config.resampler.dim,
        depth=config.resampler.depth,
        dim_head=config.resampler.dim_head,
        heads=config.resampler.heads,
        num_queries=config.model.vision.num_vision_tokens,
        num_dummy_tokens=config.model.vision.num_dummy_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=config.resampler.cross_attention_dim,
        ff_mult=config.resampler.ff_mult,
        magi_embedding_dim=magi_image_encoder.config.hidden_size if magi_image_encoder else None
    ).to(device).eval()

    if config.model.image_proj_model:
        print(f"正在从 {config.model.image_proj_model} 加载 Resampler 权重...")
        load_ckpt(resampler, config.model.image_proj_model)
    
    planner = LayoutPlanner(
        encoder_cfg={**config.model.encoder, "layout_types": config.layout_types, "max_elements": config.dataset.max_elements, "max_characters": config.dataset.max_characters},
        heads_cfg={"num_panel_classes": len(config.panel_shapes), "num_dialog_shapes": len(config.bubble_shapes)}
    ).to(device).eval()
    
    print(f"正在从 {ckpt_path} 加载 LayoutPlanner 权重...")
    state = torch.load(ckpt_path, map_location=device)
    planner.load_state_dict(state.get("model", state))
    print("模型加载完成。")

    # 2. 读取测试数据
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_list = json.load(f)
        
    results = []
    for sample_data in test_list:
        sample = DictConfig(sample_data)
        print(f"正在处理: {sample.image_path}...")
        W, H = sample.width, sample.height
        
        # 3. 构建完整的输入 Batch
        num_panels = sample.get("num_panels", 0)
        num_dialogs = sample.get("num_dialogs", 0)
        num_chars = sample.get("num_characters", 0)

        et, ei, pidx = build_element_sequence(num_panels, num_dialogs, num_chars, config.layout_types)
        
        # 视觉特征
        vis_embeds_sampled, ip_char_ids = process_visual_features(sample.get("characters", []), config, image_encoder, image_processor, magi_image_encoder, resampler, device)
        
        # 文本特征
        # captions = sample.get("panel_captions", [""] * num_panels)
        captions = sample.get("panel_captions", [""] * sample.get("num_panels", 0))
        # 确保 captions 是 List[str]
        if not isinstance(captions, list) or not all(isinstance(c, str) for c in captions):
            # 如果是 list 但里面不是 str，则尝试转换
            captions = [str(c) for c in captions]

        inputs = tokenizer(captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            txt_emb = text_encoder(**inputs).last_hidden_state
        panel_caption_embeddings = txt_emb.unsqueeze(0)

        # 构建最终对齐的视觉嵌入
        max_characters = config.dataset.max_characters
        character_ids_all = torch.tensor([[c.id for c in sorted(sample.get("characters", []), key=lambda x:x.id)] + [-1]*(max_characters - num_chars)], device=device)
        
        character_visual_embeddings = torch.zeros(1, max_characters, vis_embeds_sampled.shape[-1], device=device)
        if ip_char_ids is not None:
            for b in range(1): # Batch size is 1
                valid_ids = ip_char_ids[b]
                for t in range(max_characters):
                    token_id = character_ids_all[b, t].item()
                    if token_id != -1 and token_id in valid_ids:
                        idx = (valid_ids == token_id).nonzero(as_tuple=True)[0][0].item()
                        character_visual_embeddings[b, t] = vis_embeds_sampled[b, idx]
        
        # 创建占位符
        total_elements = len(et)
        placeholder_lidx, placeholder_sidx = [-1] * total_elements, [-1] * total_elements

        batch = {
            "element_types": torch.tensor([et], device=device),
            "element_indices": torch.tensor([ei], device=device),
            "parent_panel_indices": torch.tensor([pidx], device=device),
            "element_local_indices": torch.tensor([placeholder_lidx], device=device),
            "dialog_speaker_ids": torch.tensor([placeholder_sidx], device=device),
            "style_vector": torch.tensor([list(sample.style_parameters.values())], dtype=torch.float32, device=device),
            "aspect_ratios": torch.tensor([W / H], dtype=torch.float32, device=device),
            "panel_caption_embeddings": panel_caption_embeddings,
            "character_visual_embeddings": character_visual_embeddings,
            "character_ids": character_ids_all,
        }
        
        # 4. 模型前向传播
        with torch.no_grad():
            outputs = planner(batch)
            
        # 5. 后处理与可视化
        # (此处省略与上一版相同的后处理和可视化代码)
        id2panel = {v.id: k for k, v in config.panel_shapes.items()}
        panels, characters, dialogs = [], [], []
        p_bbox, p_offsets, p_logits = tensor_to_list(outputs["panel_bbox"][0]), tensor_to_list(outputs["panel_offsets"][0]), tensor_to_list(outputs["panel_class_logits"][0])
        for i in range(len(p_bbox)):
            base_xyxy = cxywh_to_xyxy_pixels(p_bbox[i], W, H); four_pts = offsets_to_four_points(base_xyxy, p_offsets[i], W, H); cls_id = p_logits[i].index(max(p_logits[i]))
            panels.append({"bbox_xyxy": base_xyxy, "four_points": four_pts, "panel_class_name": id2panel.get(cls_id, "unknown")})
        c_bbox = tensor_to_list(outputs["character_bbox"][0])
        for i in range(len(c_bbox)): characters.append({"bbox_xyxy": cxywh_to_xyxy_pixels(c_bbox[i], W, H)})
        d_bbox = tensor_to_list(outputs["dialog_bbox"][0])
        for i in range(len(d_bbox)): dialogs.append({"bbox_xyxy": cxywh_to_xyxy_pixels(d_bbox[i], W, H)})
            
        results.append({"image_path": sample.image_path, "width": W, "height": H, "panels": panels, "characters": characters, "dialogs": dialogs})
        
        vis_image_path = os.path.join(image_dir, sample.image_path)
        if os.path.exists(vis_image_path):
            img = Image.open(vis_image_path).convert("RGB"); draw = ImageDraw.Draw(img)
            for p in panels: draw.polygon([tuple(pt) for pt in p["four_points"]], outline=(255, 0, 0), width=3)
            for c in characters: draw.rectangle(c["bbox_xyxy"], outline=(0, 255, 0), width=3)
            for d in dialogs: draw.rectangle(d["bbox_xyxy"], outline=(0, 0, 255), width=3)
            out_vis_path = os.path.join(out_dir, f"vis_{Path(sample.image_path).stem}.png")
            img.save(out_vis_path)
            print(f"  可视化结果已保存至: {out_vis_path}")

    # 6. 保存所有结果
    out_json_path = os.path.join(out_dir, "inference_results.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n所有推理结果已保存至: {out_json_path}")

# =================================================================================
# 程序入口
# =================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayoutPlanner 推理脚本 (与训练逻辑对齐)")
    parser.add_argument("--config", required=True, help="指向 infer.yaml 的路径")
    parser.add_argument("--checkpoint", required=True, help="指向 planner 模型权重 (.pt) 的路径")
    parser.add_argument("--test_json", required=True, help="指向 test.json 输入文件的路径")
    parser.add_argument("--image_dir", required=True, help="图片文件所在的根目录")
    parser.add_argument("--output_dir", required=True, help="保存输出结果的目录")
    args = parser.parse_args()
    
    config = load_config(args.config)
    infer_and_save(config, args.checkpoint, args.test_json, args.image_dir, args.output_dir)