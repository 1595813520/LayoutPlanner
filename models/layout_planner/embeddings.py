# (在 models/layout_planner/embeddings.py 中)
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.model.d_model
        max_elements = config.dataset.max_elements
        
        # --- 离散类别嵌入 ---
        # 元素类型: 0=PAD, 1=PAGE_CTRL, 2=PANEL, 3=CHARACTER, 4=DIALOG
        self.element_type_embedding = nn.Embedding(5, d_model)
        
        # 元素索引 (用于区分 PANEL_1, PANEL_2, ...)
        # 我们为所有类型的元素共享一个位置索引嵌入层
        self.position_idx_embedding = nn.Embedding(max_elements, d_model)

        # --- 连续/高维向量映射 ---
        # 1. 风格向量 (5维) -> d_model
        self.style_mlp = nn.Sequential(
            nn.Linear(5, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # 2. 文本嵌入 (假设来自CLIP，维度为768) -> d_model
        # 为页面标题和Panel标题共享一个映射层
        self.text_embed_projection = nn.Linear(768, d_model)
        
        # 3. (可选) 重要性权重 (1维) -> d_model
        self.importance_projection = nn.Linear(1, d_model)
        
        # --- 最终的序列位置编码 ---
        # 在所有嵌入相加后，再加入一个全局的序列位置编码
        self.final_positional_encoding = nn.Embedding(max_elements, d_model)
        
    
    def forward(self, batch):
        # 从batch中解包输入
        element_types = batch['element_types'] # (B, SeqLen)
        element_indices = batch['element_indices'] # (B, SeqLen)
        style_vectors = batch['style_parameters'] # (B, 5)
        page_captions = batch['page_caption_embeds'] # (B, 768)
        panel_captions = batch['panel_caption_embeds'] # (B, NumPanels, 768)
        # ... 其他输入 ...
        
        B, SeqLen = element_types.shape
        final_embeddings = torch.zeros(B, SeqLen, self.d_model, device=element_types.device)

        # 1. 基础类型嵌入
        final_embeddings += self.element_type_embedding(element_types)
        
        # 2. 索引嵌入
        final_embeddings += self.position_idx_embedding(element_indices)

        # 3. 根据类型加入特定信息
        for i in range(SeqLen):
            # 获取当前位置所有batch样本的类型
            current_types = element_types[:, i]
            
            # 如果是PAGE_CTRL Token (假设类型为1)
            if torch.any(current_types == 1):
                mask = (current_types == 1)
                # 注入风格嵌入
                style_embed = self.style_mlp(style_vectors) # (B, d_model)
                final_embeddings[mask, i] += style_embed[mask]
                # 注入页面标题嵌入
                page_caption_embed = self.text_embed_projection(page_captions) # (B, d_model)
                final_embeddings[mask, i] += page_caption_embed[mask]
                
            # 如果是PANEL Token (假设类型为2)
            if torch.any(current_types == 2):
                mask = (current_types == 2)
                # 注入Panel标题嵌入 (需要根据panel索引匹配)
                # 此处逻辑需要仔细实现，将(B, NumPanels, 768)的数据对应到序列中的正确位置
                # ...
                
        # 4. 加入最终的全局位置编码
        # 创建一个从0到SeqLen-1的位置ID序列
        position_ids = torch.arange(SeqLen, dtype=torch.long, device=element_types.device).unsqueeze(0).expand_as(element_types)
        final_embeddings += self.final_positional_encoding(position_ids)
        
        # 5. 生成padding mask
        # padding_mask为True的地方表示是padding，在Attention中会被忽略
        padding_mask = (element_types == 0) # 假设0是PAD_TOKEN的类型
        
        return final_embeddings, padding_mask