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
        )

    @torch.no_grad()
    def _build_parent_remap(self, etypes_1d, eindices_1d):
        """
        返回: dict[orig_panel_idx] -> kept_panel_pos (0..num_kept-1)
        kept_panel_pos 的定义：在当前样本序列中的 panel token 顺序位置（压缩后）
        """
        panel_positions = torch.nonzero(etypes_1d == self.layout_types['TYPE_PANEL'], as_tuple=False).flatten()
        remap = {}
        k = 0
        for pos in panel_positions.tolist():
            orig = int(eindices_1d[pos].item())  # tokenizer里 panel 的 element_indices = 原始 panel 顺序
            remap[orig] = k
            k += 1
        return remap

    def forward(self, batch):
        """
        batch keys:
          element_types (B,S), element_indices (B,S), parent_panel_indices (B,S), style_vector (B,4)
        返回 list[ (panel_out, dialog_out, char_out) ]，每个样本一组，方便 loss 做样本内裁剪匹配
        """
        etypes = batch["element_types"]       # (B,S)
        eidxs  = batch["element_indices"]     # (B,S)
        pidxs  = batch["parent_panel_indices"]# (B,S)
        style  = batch["style_vector"]        # (B,4)

        enc = self.encoder(etypes, eidxs, pidxs, style)
        seq_feats = enc["seq_feats"]          # (B,S,D)

        outputs = []
        B, S, D = seq_feats.shape
        for b in range(B):
            feats = seq_feats[b]              # (S,D)
            et_b  = etypes[b]                 # (S,)
            pi_b  = pidxs[b]                  # (S,)
            # 构建 orig_panel_idx -> kept_panel_pos
            remap = self._build_parent_remap(et_b, eidxs[b])
            # 生成 parent_pos 向量，仅对 dialog/char 有意义
            parent_pos = torch.full_like(pi_b, fill_value=-1)
            for i in range(S):
                if et_b[i].item() in (self.layout_types['TYPE_DIALOG'], self.layout_types['TYPE_CHAR']):
                    orig = int(pi_b[i].item())
                    parent_pos[i] = remap.get(orig, -1)  # 若被截断或非法 -> -1

            # Heads 需要的输入是整个序列，但内部会按 mask 取不同元素
            p_out, d_out, c_out = self.heads(
                feats, et_b, parent_pos
            )
            outputs.append((p_out, d_out, c_out))
        return outputs
