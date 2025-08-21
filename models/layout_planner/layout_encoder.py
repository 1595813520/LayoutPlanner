# DiffSensei-main/layout-generator/models/layout_planner/layout_encoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFP32(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).to(x.dtype)

class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width*4)
        self.fc2 = nn.Linear(width*4, width)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MultiheadSelfAttn(nn.Module):
    def __init__(self, n_ctx, width, heads, dropout=0.0):
        super().__init__()
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.heads = heads
        self.dropout = dropout

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, L, C)
        key_padding_mask: (B, L) boolean (True for PAD positions) or None

        Returns: (B, L, C)
        """
        B, L, C = x.shape
        assert C % self.heads == 0, "d_model must be divisible by num_heads"
        head_dim = C // self.heads

        # produce qkv and reshape to (B, heads, L, head_dim)
        qkv = self.c_qkv(x)  # (B, L, 3*C)
        qkv = qkv.view(B, L, 3, self.heads, head_dim)  # (B, L, 3, heads, head_dim)

        # reorder to (3, B, heads, L, head_dim) then split
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, heads, L, head_dim)

        # attention: (B, heads, L_q, L_k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

        # key_padding_mask: expect shape (B, L_k) (bool or 0/1)
        if key_padding_mask is not None:
            # ensure boolean and correct device
            mask = key_padding_mask.to(torch.bool)
            if mask.dim() != 2 or mask.shape[0] != B or mask.shape[1] != L:
                # try to adapt if L differs: if mask shorter/longer, broadcast or slice accordingly
                # but best is to raise informative error
                raise RuntimeError(f"key_padding_mask shape {mask.shape} incompatible with x shape (B={B}, L={L})")
            # mask needs to be (B, 1, 1, L) to broadcast to att (B, heads, L, L)
            att = att.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        att = torch.softmax(att, dim=-1)
        if self.dropout and self.training:
            att = F.dropout(att, p=self.dropout)

        # att @ v -> (B, heads, L, head_dim)
        out = (att @ v)

        # rearrange back to (B, L, C)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        out = self.c_proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, n_ctx, width, heads, dropout=0.0):
        super().__init__()
        self.ln1 = LayerNormFP32(width)
        self.attn = MultiheadSelfAttn(n_ctx, width, heads, dropout)
        self.ln2 = LayerNormFP32(width)
        self.mlp = MLP(width)
    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TokenLayoutEncoder(nn.Module):
    """
    直接吃:
      - element_types (B,S)
      - element_indices (B,S)
      - parent_panel_indices (B,S)  对于 panel/page = -1; dialog/char = [0..#panels-1] 或 -1
      - style_vector (B,4)          只加到 PAGE token 上
    输出:
      - seq_feats (B,S,d_model)     给 heads 用
      - key_padding_mask (B,S)      供可选的 attention mask
    """
    def __init__(self,
                 max_elements: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 layout_types,
                 caption_embed_dim: int = 768,
                 use_positional_encoding: bool = True,
                 use_final_ln: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.max_elements = max_elements
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.use_final_ln = use_final_ln
        self.layout_types = layout_types

        # token embeddings
        self.type_embed = nn.Embedding(5, d_model)                 # PAD/PAGE/PANEL/CHAR/DIALOG
        self.index_embed = nn.Embedding(max_elements, d_model)     # element_indices
        self.parent_bucket = nn.Embedding(max_elements+2, d_model) # -1 -> bucket 0, else +1 shift

        # style to d_model
        self.style_mlp = nn.Sequential(
            nn.Linear(4, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.caption_proj = nn.Linear(caption_embed_dim, d_model)

        if use_positional_encoding:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_elements, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(max_elements, d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.final_ln = LayerNormFP32(d_model) if use_final_ln else nn.Identity()

    @staticmethod
    def _bucket_parent(parent_idx: torch.Tensor, max_elements: int):
        # parent=-1 -> 0, else parent+2 (留出 1 位置备扩展)
        # clamp 到 [0, max_elements+1]
        p = parent_idx.clone()
        p = torch.where(p < 0, torch.zeros_like(p), p + 2)
        return torch.clamp(p, 0, max_elements+1)

    def forward(self, element_types, element_indices, parent_panel_indices, style_vector, panel_caption_embeddings=None):
        """
        element_types: (B,S) long
        element_indices: (B,S) long
        parent_panel_indices: (B,S) long
        style_vector: (B,4) float
        """
        B, S = element_types.shape
        assert S <= self.max_elements, f"S={S} > max_elements={self.max_elements}"

        # 1. Base embed (type + index + parent)
        x = self.type_embed(element_types) + self.index_embed(torch.clamp(element_indices, 0, self.max_elements-1))
        # parent embed
        p_bucket = self._bucket_parent(parent_panel_indices, self.max_elements)  # (B,S)
        x = x + self.parent_bucket(p_bucket)

        # 2 inject style ONLY to PAGE tokens
        is_page = (element_types == self.layout_types['TYPE_PAGE']).unsqueeze(-1)  # (B,S,1)
        style_e = self.style_mlp(style_vector).unsqueeze(1)   # (B,1,D)
        x = x + style_e * is_page
        
        # 3 Inject caption embeddings ONLY to PANEL tokens ---
        if panel_caption_embeddings is not None:
            is_panel = (element_types == self.layout_types['TYPE_PANEL']) # (B, S)
            
            # Project caption embeddings to d_model
            caption_e = self.caption_proj(panel_caption_embeddings) # (B, NumPanels, D)
            
            # Create a zero tensor to "scatter" the caption embeddings into
            caption_injection = torch.zeros_like(x)
            
            # Get the indices of the panels within their own group (0, 1, 2...)
            panel_indices = element_indices[is_panel] 
            
            # Create batch indices to correctly select from the caption tensor
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, S)[is_panel]
            
            # Select the correct caption embedding for each panel token
            selected_captions = caption_e[batch_indices, panel_indices]
            
            # Place the selected caption embeddings into the correct sequence positions
            caption_injection[is_panel] = selected_captions
            
            # Add the caption embeddings to the base embeddings
            x = x + caption_injection

        # 4. Add positional encoding
        if self.use_positional_encoding:
            x = x + self.pos_embed[:, :S, :]
            
        # 5. Create padding mask and pass through Transformer blocks
        key_padding_mask = (element_types == self.layout_types['TYPE_PAD'])
        for blk in self.blocks:
            x = blk(x, key_padding_mask)

        x = self.final_ln(x)
        return {"seq_feats": x, "key_padding_mask": key_padding_mask}
