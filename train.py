# train.py
from datetime import datetime
import os
import time
import argparse
from omegaconf import OmegaConf
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, AutoModel, CLIPTextModelWithProjection
from src.datasets import MangaLayoutDataset, collate_fn
from src.utils import load_ckpt, mean_multiple_ip_embeds
from models.layout_planner.planner import LayoutPlanner
from models.layout_planner.resampler import Resampler
from src.losses import LayoutCompositeLoss

def seed_everything(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def apply_style_cfg_dropout(style_vectors, p):
    """
    style_vectors: Tensor (B,4)
    p: dropout prob (0..1)
    For each sample with prob p, zero out style vector (unconditional).
    """
    if p <= 0.0:
        return style_vectors
    device = style_vectors.device
    B = style_vectors.shape[0]
    mask = (torch.rand(B, device=device) >= p).float().unsqueeze(-1)  # 1 means keep, 0 means drop
    return style_vectors * mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="planner.yaml")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--resume_log_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_conf = OmegaConf.create(args_dict)
    config = OmegaConf.merge(config, args_conf)
    
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载文本编码器和分词器
    tokenizer = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder_2")
    text_encoder.to(device)
    text_encoder_2.to(device)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    image_encoder_path = config.model.image_encoder_path
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    image_encoder.to(device)
    image_encoder.requires_grad_(False)
    if config.model.magi_image_encoder_path is not None:
        magi_image_encoder = AutoModel.from_pretrained(config.model.magi_image_encoder_path, trust_remote_code=True).crop_embedding_model
        magi_image_encoder.to(device)
        magi_image_encoder.requires_grad_(False)
    else:
        magi_image_encoder = None
    
    
    # Init adapter modules
    image_proj_model = Resampler(
        dim=config.resampler.dim,
        depth=config.resampler.depth,
        dim_head=config.resampler.dim_head,
        heads=config.resampler.heads,
        num_queries=config.model.vision.num_vision_tokens,
        num_dummy_tokens=config.model.vision.num_dummy_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=config.resampler.cross_attention_dim,
        ff_mult=config.resampler.ff_mult,
        magi_embedding_dim=magi_image_encoder.config.hidden_size if magi_image_encoder is not None else None
        # use_magi=config.model.magi_image_encoder_path is not None
    ).to(device)
    
    if config.model.image_proj_model is not None:
        load_ckpt(image_proj_model, config.model.image_proj_model)
        
    # Dataset & Loader
    dataset = MangaLayoutDataset(
        ann_source=config.data.annotation_path,
        image_dir=config.data.image_dir,
        config=config,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        max_panels=config.dataset.max_panels,
        max_num_ips=config.model.vision.num_ips,
        max_num_ip_sources=config.model.vision.num_ip_sources,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.dataset.num_workers,
        collate_fn=lambda b: collate_fn(b, config),
        pin_memory=True,
    )
    
    planner = LayoutPlanner(
        encoder_cfg={
            "max_elements": config.dataset.max_elements,
            "max_characters": config.dataset.max_characters,
            "d_model": config.model.encoder.d_model,
            "num_layers": config.model.encoder.num_layers,
            "num_heads": config.model.encoder.num_heads,
            "use_positional_encoding": config.model.encoder.use_positional_encoding,
            "use_final_ln": config.model.encoder.use_final_ln,
            "dropout": config.model.encoder.dropout,
            "layout_types": config.layout_types
        },
        heads_cfg={
            "num_panel_classes": len(config.panel_shapes),
            "num_dialog_shapes": len(config.bubble_shapes),  # 不预测 bubble shape
            "layout_types": config.layout_types
        }
    ).to(device)
    
    # loss & optimizer
    criterion = LayoutCompositeLoss(
        lambda_style=config.training.lambda_style,
        lambda_geom=config.training.lambda_geom
    ).to(device)
    
    lr = float(config.training.lr)
    weight_decay = float(config.training.weight_decay)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=lr, weight_decay=weight_decay)
    style_dropout_p = float(config.training.style_cfg_dropout)
    epochs = int(config.training.epochs)
    save_dir = config.training.save_dir
    os.makedirs(save_dir, exist_ok=True)
    clip_grad = float(config.training.clip_grad_norm)
    best_loss = float("inf")
    global_step = 0
    
    for epoch in range(1, epochs+1):
        planner.train()
        running = 0.0
        t0 = time.time()
        for it, batch in enumerate(loader):
            # ==== 1. 直接用collate_fn的输出，无需再二次单条处理 ====
            # 自动to(device)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            # (可选) panel captions嵌入（批处理，padding已在collate_fn完成）
            captions_nested = batch["panel_captions"]
            flat_captions = [c for caps in captions_nested for c in caps]
            if flat_captions:
                inputs = tokenizer(flat_captions, padding=True, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    emb = text_encoder(**inputs).last_hidden_state[:, 0, :]
                per_sample_embeds = []
                idx = 0
                for caps in captions_nested:
                    n = len(caps)
                    if n > 0:
                        per_sample_embeds.append(emb[idx:idx+n])
                    else:
                        per_sample_embeds.append(torch.zeros((0, emb.shape[-1]), device=device))
                    idx += n
            else:
                per_sample_embeds = [torch.zeros((0, text_encoder.config.hidden_size), device=device) for _ in captions_nested]
            # 按 config 固定长度 pad 到 max_panels
            max_panels = config.dataset.max_panels
            D = text_encoder.config.hidden_size
            padded_embeds = []
            for e in per_sample_embeds:
                if e.shape[0] < max_panels:
                    pad = torch.zeros((max_panels - e.shape[0], D), device=device)
                    e = torch.cat([e, pad], dim=0)
                elif e.shape[0] > max_panels:
                    e = e[:max_panels]
                padded_embeds.append(e)
            batch["panel_caption_embeddings"] = torch.stack(padded_embeds, dim=0)
            
            # ==== 2. style cfg dropout ====
            batch["style_vector"] = apply_style_cfg_dropout(batch["style_vector"], style_dropout_p)
            
            # ==== 3. Encode image features + Resampler =====
            B = batch["ip_images"].shape[0]
            N_ips = config.model.vision.num_ips
            N_src = config.model.vision.num_ip_sources

            with torch.no_grad():
                # 展平成[batch*角色数*ip_source, 3, 224, 224]
                ip_images_flat = batch["ip_images"].view(B * N_ips * N_src, *batch["ip_images"].shape[2:])
                
                # ——改用 last_hidden_state！——
                outputs = image_encoder(ip_images_flat.to(device), output_hidden_states=False, return_dict=True)
                image_embeds_raw = outputs.last_hidden_state  # [16, 257, 1280]
                
                # reshape为 [B, N_ips, N_src, seq_len, D_img]
                image_embeds_raw = image_embeds_raw.view(B, N_ips, N_src, image_embeds_raw.shape[1], image_embeds_raw.shape[2])
                # 转置为 [B, N_src, N_ips, seq_len, D_img]
                image_embeds_raw = image_embeds_raw.transpose(1, 2).contiguous()
                # 最终展平为 [B * N_src, N_ips, seq_len, D_img]
                image_embeds_raw = image_embeds_raw.view(B * N_src, N_ips, image_embeds_raw.shape[-2], image_embeds_raw.shape[-1])

                # 处理 MAGI 部分同理
                if magi_image_encoder is not None:
                    magi_images_flat = batch["magi_ip_images"].view(B * N_ips * N_src, *batch["magi_ip_images"].shape[2:])
                    magi_hidden = magi_image_encoder(magi_images_flat.to(device, dtype=torch.float32)).last_hidden_state
                    magi_embeds = magi_hidden[:, 0]
                    magi_embeds = magi_embeds.view(B, N_ips, N_src, -1).transpose(1, 2)
                    magi_image_embeds = magi_embeds.contiguous().view(B * N_src, N_ips, -1).to(dtype=torch.float32)
                else:
                    magi_image_embeds = None


            image_embeds_all = image_proj_model(image_embeds_raw, magi_image_embeds)
            image_embeds_final = mean_multiple_ip_embeds(image_embeds_all, batch["ip_exists"], config, B)
            vis_tokens_per_char = image_embeds_final[:, config.model.vision.num_dummy_tokens:, :]
            D_cross = vis_tokens_per_char.shape[-1]
            num_vision_tokens = config.model.vision.num_vision_tokens
            # visual features
            character_visual_tokens = vis_tokens_per_char.view(B, N_ips, num_vision_tokens, D_cross)
            character_visual_embeddings_sampled = character_visual_tokens.mean(dim=2)  # (B, N_ips, D)
            ip_char_ids = batch["ip_char_ids"]   # (B, N_ips)
            character_ids = batch["character_ids"]  # (B, max_characters)
            # 对齐角色ID，得到(B, max_characters, D)的完整visual特征
            max_characters = character_ids.shape[1]
            D = character_visual_embeddings_sampled.shape[-1]
            device = character_visual_embeddings_sampled.device
            character_visual_embeddings = torch.zeros(B, max_characters, D, device=device)

            for b in range(B):
                for t in range(max_characters):
                    token_id = character_ids[b, t].item()
                    # id可能为pad(-1或0)，需统一处理；这里用-1和ip_char_ids里pad一致
                    if token_id > 0 and token_id in ip_char_ids[b]:
                        idx = (ip_char_ids[b] == token_id).nonzero(as_tuple=False)[0].item()  # 第一个匹配
                        character_visual_embeddings[b, t] = character_visual_embeddings_sampled[b, idx]
                    # else 保持全零
            batch["character_visual_embeddings"] = character_visual_embeddings
            
            # ==== 4. 前向、loss、step ====
            outputs = planner(batch)  # list of per-sample outputs
            loss, logs = criterion(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(planner.parameters(), clip_grad)
            optimizer.step()
            running += float(loss.item())
            global_step += 1
            if (it + 1) % 200 == 0:
                avg = running / (it + 1)
                print(f"[Epoch {epoch} | iter {it+1}/{len(loader)}] avg_loss={avg:.4f} geom={logs['geom_loss']:.4f}  pred={logs['pred_loss']:.4f} style={logs['style_loss']:.4f}")
        epoch_loss = running / max(1, len(loader))
        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s | loss={epoch_loss:.4f}")
        # save
        ckpt = os.path.join(save_dir, f"planner_epoch{epoch}.pt")
        torch.save({"model": planner.state_dict(), "epoch": epoch}, ckpt)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": planner.state_dict(), "epoch": epoch}, os.path.join(save_dir, "planner_best.pt"))
    print("Training finished.")
    print(f"Best model saved to {os.path.join(save_dir, 'planner_best.pt')}")
    
if __name__ == "__main__":
    main()