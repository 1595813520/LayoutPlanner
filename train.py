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
from src.utils import load_ckpt, mean_multiple_ip_embeds, load_planner_ckpt, encode_in_chunks, size_buckets
from models.layout_planner.planner import LayoutPlanner
from models.layout_planner.resampler import Resampler
from src.losses import LayoutCompositeLoss
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter
def apply_style_cfg_dropout(style_vectors, p, device):
    """
    style_vectors: Tensor (B,4)
    p: dropout prob (0..1)
    For each sample with prob p, zero out style vector (unconditional).
    """
    if p <= 0.0:
        return style_vectors
    B = style_vectors.shape[0]
    mask = (torch.rand(B, device=device) >= p).float().unsqueeze(-1)  # 1 means keep, 0 means drop
    return style_vectors * mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--max_train_steps", type=int, default=100000, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--resume_log_dir", type=str, default=None, help="Log dir to resume training from (must contain config.yaml and checkpoints).")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_conf = OmegaConf.create(args_dict)
    config = OmegaConf.merge(config, args_conf)
    set_seed(config.training.seed)
    # ---- 自动选择 find_unused_parameters ----
    find_unused_params = False  # 默认关闭以提高性能
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_params)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )

    # ---- 日志设置 ----
    # 1. 确定 log_dir
    if args.resume_log_dir is not None:
        log_dir = args.resume_log_dir
    else:
        log_dir = os.path.join(config.training.save_dir, "logs", datetime.now().strftime("%Y-%m%d-%H:%M"))

    # 2. 只有主进程负责创建和保存配置
    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    # 3. 同步所有进程，保证 log_dir 可用
    accelerator.wait_for_everyone()

    # 加载文本编码器和分词器
    tokenizer = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder_2")
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    image_encoder_path = config.model.image_encoder_path
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    image_encoder.requires_grad_(False)
    if config.model.magi_image_encoder_path is not None:
        magi_image_encoder = AutoModel.from_pretrained(config.model.magi_image_encoder_path, trust_remote_code=True).crop_embedding_model
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
    )
    
    if config.model.image_proj_model is not None:
        load_ckpt(image_proj_model, config.model.image_proj_model)
        image_proj_model.requires_grad_(False)
        
    # Dataset & Loader
    dataset = MangaLayoutDataset(
        ann_source=config.data.annotation_path,
        image_dir=config.data.image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        max_panels=config.dataset.max_panels,
        max_num_ips=config.model.vision.num_ips,
        max_num_ip_sources=config.model.vision.num_ip_sources,
    )
    
    # batch_sampler = BucketBatchSampler(
    #     dataset=dataset,
    #     batch_size=config.training.batch_size
    # )
    
    loader = DataLoader(
        dataset,
        # batch_sampler=batch_sampler, 
        batch_size=config.training.batch_size,
        shuffle=config.dataset.shuffle,
        num_workers=config.training.num_workers * accelerator.num_processes,
        # collate_fn=lambda b: collate_fn(b, config, device=accelerator.device), # 修改 collate 直接放 GPU
        collate_fn=lambda b: collate_fn(b, config), 
        pin_memory=True,
        persistent_workers=True
    )
    
    # for i, batch in enumerate(loader):
    #     for k,v in batch.items():
    #         if torch.is_tensor(v):
    #             batch[k] = v.to(accelerator.device, non_blocking=True)
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
    )
    
    # loss & optimizer
    criterion = LayoutCompositeLoss(
        lambda_style=config.training.lambda_style,
        lambda_geom=config.training.lambda_geom,
        style_mu=config.training.style_mu, 
        style_sigma=config.training.style_sigma,
        rect_class_id=config.panel_shapes["panel_rect"].id   
    )
        
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    image_proj_model.to(accelerator.device, dtype=weight_dtype)
    if magi_image_encoder is not None:
        magi_image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    lr = float(config.training.lr)
    weight_decay = float(config.training.weight_decay)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Prepare everything with accelerator    
    planner, criterion, optimizer, loader = accelerator.prepare(
        planner, criterion, optimizer, loader
    )
    
    style_dropout_p = float(config.training.style_cfg_dropout)
    # epochs = int(config.training.epochs)
    save_dir = config.training.save_dir
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    clip_grad = float(config.training.clip_grad_norm)
    best_loss = float("inf")
    step_in_epoch = 0
    running = 0.0
    
    if args.resume_log_dir is not None:
        ckpt_files = [f for f in os.listdir(log_dir) if f.startswith("planner_step") and f.endswith(".pt")]
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {log_dir}")
        last_ckpt_file = sorted(ckpt_files, key=lambda x: int(x.split("planner_step")[1].split(".")[0]))[-1]
        ckpt_path = os.path.join(log_dir, last_ckpt_file)
        global_step = load_planner_ckpt(planner, optimizer, ckpt_path, map_location=accelerator.device)
    else:
        global_step = 0
    
    while global_step < config.training.max_train_steps:
    # for epoch in range(1, epochs+1):
        planner.train()
        t0 = time.time()
        for i, batch in enumerate(loader):
        # for batch in loader:
            # (可选) panel captions嵌入（批处理，padding已在collate_fn完成）
            captions_nested = batch["panel_captions"]
            flat_captions = [c for caps in captions_nested for c in caps]
            if flat_captions:
                inputs = tokenizer(flat_captions, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                with torch.no_grad():
                    emb = text_encoder(**inputs).last_hidden_state[:, 0, :]
                per_sample_embeds = []
                idx = 0
                for caps in captions_nested:
                    n = len(caps)
                    if n > 0:
                        per_sample_embeds.append(emb[idx:idx+n])
                    else:
                        per_sample_embeds.append(torch.zeros((0, emb.shape[-1]), device=accelerator.device))
                    idx += n
            else:
                per_sample_embeds = [torch.zeros((0, text_encoder.config.hidden_size), device=accelerator.device) for _ in captions_nested]
            # 按 config 固定长度 pad 到 max_panels
            max_panels = config.dataset.max_panels
            D = text_encoder.config.hidden_size
            padded_embeds = []
            for e in per_sample_embeds:
                if e.shape[0] < max_panels:
                    pad = torch.zeros((max_panels - e.shape[0], D), device=accelerator.device)
                    e = torch.cat([e, pad], dim=0)
                elif e.shape[0] > max_panels:
                    e = e[:max_panels]
                padded_embeds.append(e)
            batch["panel_caption_embeddings"] = torch.stack(padded_embeds, dim=0)
            
            # ==== 2. style cfg dropout ====
            batch["style_vector"] = apply_style_cfg_dropout(batch["style_vector"], style_dropout_p, accelerator.device)
            
            # ==== 3. Encode image features + Resampler =====
            B = batch["ip_images"].shape[0]
            N_ips = config.model.vision.num_ips
            N_src = config.model.vision.num_ip_sources

            with torch.no_grad():
                # 展平成[batch*角色数*ip_source, 3, 224, 224]
                ip_images_flat = batch["ip_images"].view(B * N_ips * N_src, *batch["ip_images"].shape[2:])
                image_embeds_raw = encode_in_chunks(image_encoder, ip_images_flat,
                                    chunk_size=config.training.batch_size,
                                    device=accelerator.device,
                                    dtype=weight_dtype)
                # # ——改用 last_hidden_state！——
                # outputs = image_encoder(ip_images_flat.to(accelerator.device), output_hidden_states=False, return_dict=True)
                # image_embeds_raw = outputs.last_hidden_state  # [16, 257, 1280]
                
                # reshape为 [B, N_ips, N_src, seq_len, D_img]
                image_embeds_raw = image_embeds_raw.view(B, N_ips, N_src, image_embeds_raw.shape[1], image_embeds_raw.shape[2])
                # 转置为 [B, N_src, N_ips, seq_len, D_img]
                image_embeds_raw = image_embeds_raw.transpose(1, 2).contiguous()
                # 最终展平为 [B * N_src, N_ips, seq_len, D_img]
                image_embeds_raw = image_embeds_raw.view(B * N_src, N_ips, image_embeds_raw.shape[-2], image_embeds_raw.shape[-1])

                # 处理 MAGI 部分同理
                if magi_image_encoder is not None:
                    magi_images_flat = batch["magi_ip_images"].view(B * N_ips * N_src, *batch["magi_ip_images"].shape[2:])
                    magi_hidden = encode_in_chunks(magi_image_encoder, magi_images_flat,
                                chunk_size=config.training.batch_size,
                                device=accelerator.device,
                                dtype=weight_dtype)
                    # magi_hidden = magi_image_encoder(magi_images_flat.to(accelerator.device, dtype=weight_dtype)).last_hidden_state
                    # magi 用 [CLS] 向量
                    magi_embeds = magi_hidden[:, 0]
                    magi_embeds = magi_embeds.view(B, N_ips, N_src, -1).transpose(1, 2)
                    magi_image_embeds = magi_embeds.contiguous().view(B * N_src, N_ips, -1).to(dtype=weight_dtype)
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
            character_visual_embeddings = torch.zeros(B, max_characters, D, device=accelerator.device)

            # real_ids = ip_char_ids[b][:N_ips].tolist()
            for b in range(B):
                real_n = character_visual_embeddings_sampled.shape[1] if character_visual_embeddings_sampled.ndim == 3 else character_visual_embeddings_sampled[b].shape[0]
                # 有的实现N_ips可落在维度1，有的在dim0
                valid_ids = ip_char_ids[b][:real_n]  # 严格只看有embedding的id
                for t in range(max_characters):
                    token_id = character_ids[b, t].item()
                    if token_id > 0 and token_id in valid_ids:
                        idxs = (valid_ids == token_id).nonzero(as_tuple=False)
                        if idxs.numel() > 0:
                            idx = idxs[0].item()
                            if idx < real_n:
                                character_visual_embeddings[b, t] = character_visual_embeddings_sampled[b, idx]
                    # else 保持全零
            batch["character_visual_embeddings"] = character_visual_embeddings
            
            # ==== 4. 前向、loss、step ====
            with accelerator.accumulate(planner):
                with accelerator.autocast():
                    outputs = planner(batch)  # list of per-sample outputs
                    loss, logs = criterion(outputs, batch)
                accelerator.backward(loss)
                
            synced_loss = accelerator.gather(loss).mean()
            running += synced_loss.item()

            if clip_grad is not None and global_step % config.training.grad_clip_interval == 0:
                accelerator.clip_grad_norm_(planner.parameters(), config.training.clip_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_in_epoch += 1
            
            if accelerator.is_main_process and (global_step % 100 == 0):
                avg = running / max(1, step_in_epoch)
                print(f"[global_step {global_step}] avg_loss={avg:.4f} geom={logs['geom_loss']:.4f}  pred={logs['pred_loss']:.4f} style={logs['style_loss']:.4f}")

            # ---- TensorBoard logging ----
            log_interval = int(config.training.log_interval)
            if accelerator.is_main_process and global_step % log_interval == 0:
                avg_loss = running / log_interval
                writer.add_scalar("Loss/total", avg_loss, global_step)
                writer.add_scalar("Loss/pred", logs["pred_loss"], global_step)
                writer.add_scalar("Loss/geom", logs["geom_loss"], global_step)
                writer.add_scalar("Loss/style", logs["style_loss"], global_step)
            
            save_interval = int(config.training.save_interval)
            # 按一定步数保存模型
            if accelerator.is_main_process and (global_step % save_interval == 0):
                epoch_loss = running / max(1, step_in_epoch)
                print(f"[global_step {global_step}] done in {time.time()-t0:.1f}s | loss={epoch_loss:.4f}")
                ckpt = os.path.join(save_dir, f"planner_step{global_step}.pt")
                accelerator.save({
                    "model": accelerator.get_state_dict(planner),  # 兼容 DDP
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step
                }, ckpt)
                print(f"model saved to {ckpt}")
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    accelerator.save({
                        "model": accelerator.get_state_dict(planner),
                        "global_step": global_step,
                        "best_loss": best_loss
                    }, os.path.join(save_dir, "planner_best.pt"))
            if global_step >= config.training.max_train_steps:
                break  

        running = 0.0
        step_in_epoch = 0
    
    if accelerator.is_main_process:
        print("Training finished.")
        print(f"Best model saved to {os.path.join(save_dir, 'planner_step{global_step}.pt')}")
    
if __name__ == "__main__":
    main()