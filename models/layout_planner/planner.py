# DiffSensei-main/layout-generator/models/layout_planner/planner.py
import torch
import torch.nn as nn
from .layout_encoder import TokenLayoutEncoder
from .heads import ParallelPredictionHeads

class LayoutPlanner(nn.Module):
    def __init__(self, encoder_cfg, heads_cfg):
        super().__init__()
        self.layout_types = encoder_cfg["layout_types"]
        self.encoder = TokenLayoutEncoder(**encoder_cfg)
        self.heads = ParallelPredictionHeads(
            d_model=encoder_cfg["d_model"],
            num_panel_classes=heads_cfg.get("num_panel_classes", 4),
            num_dialog_shapes=heads_cfg.get("num_dialog_shapes", 4),
            layout_types=encoder_cfg["layout_types"]
        )

    def forward(self, batch):
        """
        batch keys:
          element_types (B,S), element_indices (B,S), 
          parent_panel_indices (B,S), style_vector (B,4),
          panel_caption_embeddings (B, NumPanels, 768)
        """
        # --- 1. Encoder 前向传播 ---
        # 直接将 caption 嵌入传递给 encoder
        enc_outputs = self.encoder(
            element_types=batch["element_types"],
            element_indices=batch["element_indices"],
            element_local_indices=batch["element_local_indices"],
            parent_panel_indices=batch["parent_panel_indices"],
            style_vector=batch["style_vector"],
            aspect_ratios=batch["aspect_ratios"],   
            panel_caption_embeddings=batch["panel_caption_embeddings"],
            character_ids=batch["character_ids"],
            character_visual_embeddings=batch["character_visual_embeddings"]
        )
        seq_feats = enc_outputs["seq_feats"]  # (B, S, D)

        # --- 2. Heads 前向传播 (直接处理整个批次) ---
        predictions = self.heads(
            lfm_output=seq_feats,
            element_types=batch["element_types"],
            element_indices=batch["element_indices"], # Heads 需要这个来匹配父 panel
            parent_panel_indices=batch["parent_panel_indices"]
        )
        
        return predictions