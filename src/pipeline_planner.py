# pipeline_planner.py
import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor, AutoModel
from src.utils import load_ckpt, mean_multiple_ip_embeds
from models.layout_planner.planner import LayoutPlanner
from models.layout_planner.resampler import Resampler

def build_element_sequence(num_panels, num_dialogs, num_chars, layout_types, max_elements):
    et, ei, pidx = [layout_types.TYPE_PAGE], [0], [-1]
    for i in range(num_panels):
        et.append(layout_types.TYPE_PANEL); ei.append(i); pidx.append(-1)
    for j in range(num_dialogs):
        et.append(layout_types.TYPE_DIALOG); ei.append(j); pidx.append(-1)
    for k in range(num_chars):
        et.append(layout_types.TYPE_CHAR); ei.append(k); pidx.append(-1)
    # pad to max_elements
    while len(et) < max_elements:
        et.append(layout_types.TYPE_PAD)
        ei.append(0)
        pidx.append(-1)
    return torch.tensor([et]), torch.tensor([ei]), torch.tensor([pidx])

def load_models(cfg, device):
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.pretrained_model_path, subfolder="text_encoder").to(device).eval()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.model.image_encoder_path).to(device).eval()

    magi_image_encoder = None
    if cfg.model.magi_image_encoder_path:
        magi_image_encoder = AutoModel.from_pretrained(cfg.model.magi_image_encoder_path, trust_remote_code=True).crop_embedding_model.to(device).eval()

    resampler = Resampler(
        dim=cfg.resampler.dim,
        depth=cfg.resampler.depth,
        dim_head=cfg.resampler.dim_head,
        heads=cfg.resampler.heads,
        num_queries=cfg.model.vision.num_vision_tokens,
        num_dummy_tokens=cfg.model.vision.num_dummy_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=cfg.resampler.cross_attention_dim,
        ff_mult=cfg.resampler.ff_mult,
        magi_embedding_dim=magi_image_encoder.config.hidden_size if magi_image_encoder else None
    ).to(device).eval()
    if cfg.model.image_proj_model:
        load_ckpt(resampler, cfg.model.image_proj_model)
    planner = LayoutPlanner(
        encoder_cfg={
            "max_elements": cfg.dataset.max_elements,
            "max_characters": cfg.dataset.max_characters,
            "max_panels": cfg.dataset.max_panels,
            "max_dialogs": cfg.dataset.max_dialogs,
            "d_model": cfg.model.encoder.d_model,
            "num_layers": cfg.model.encoder.num_layers,
            "num_heads": cfg.model.encoder.num_heads,
            "use_positional_encoding": cfg.model.encoder.use_positional_encoding,
            "use_final_ln": cfg.model.encoder.use_final_ln,
            "dropout": cfg.model.encoder.dropout,
            "layout_types": cfg.layout_types
        },
        heads_cfg={
            "num_panel_classes": len(cfg.panel_shapes),
            "num_dialog_shapes": len(cfg.bubble_shapes),  # 不预测 bubble shape
            "layout_types": cfg.layout_types
        }
    ).to(device).eval()
    
    return tokenizer, text_encoder, image_encoder, magi_image_encoder, resampler, planner

def process_visual_features(characters, config, image_encoder, image_processor, magi_image_encoder, resampler, device):
    if len(characters) == 0:
        return torch.zeros(1, 0, config.resampler.cross_attention_dim, device=device), None, None

    ip_images_list = []
    ip_char_ids = []
    for c in characters:
        img = Image.open(c["ip_image_path"]).convert("RGB")
        img_tensor = image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        ip_images_list.append(img_tensor)
        ip_char_ids.append(c["id"])

    ip_images = torch.stack(ip_images_list, dim=0).unsqueeze(0).to(device)
    ip_char_ids_tensor = torch.tensor([ip_char_ids], device=device)
    ip_exists = torch.ones_like(ip_char_ids_tensor, dtype=torch.float32)  # 自动全1

    with torch.no_grad():
        B, N_chars, C, H, W = ip_images.shape
        N_src = 1
        ip_images_flat = ip_images.view(B * N_chars * N_src, C, H, W)
        outputs = image_encoder(ip_images_flat, return_dict=True)
        image_embeds_raw = outputs.last_hidden_state.view(B, N_chars, N_src, -1, outputs.last_hidden_state.shape[-1])
        image_embeds_raw = image_embeds_raw.transpose(1, 2).contiguous()
        image_embeds_raw = image_embeds_raw.view(B * N_src, N_chars, image_embeds_raw.shape[-2], image_embeds_raw.shape[-1])

        magi_embeds = None
        if magi_image_encoder is not None:
            magi_hidden = magi_image_encoder(ip_images_flat).last_hidden_state
            magi_embeds = magi_hidden[:, 0].view(B, N_chars, N_src, -1).transpose(1, 2)
            magi_embeds = magi_embeds.contiguous().view(B * N_src, N_chars, -1)

        image_embeds_all = resampler(image_embeds_raw, magi_embeds)

        image_embeds_final = mean_multiple_ip_embeds(image_embeds_all, ip_exists, config, B)
    return image_embeds_final, ip_char_ids_tensor, ip_exists