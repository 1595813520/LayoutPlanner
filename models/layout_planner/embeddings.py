import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# ==============================================================================
# 步骤 1: 数据处理函数 (Tokenization)
# 将单个JSON标注转换为模型可以处理的张量字典。
# ==============================================================================

def process_annotation_to_tensors(annotation, text_encoder, max_seq_len=128):
    """
    Processes a single annotation JSON into a dictionary of tensors for the model.

    Args:
        annotation (dict): The annotation for a single page.
        text_encoder: A mock or real text encoder (like CLIP's) that takes a string
                      and returns a tensor.
        max_seq_len (int): The maximum sequence length for padding.

    Returns:
        dict: A dictionary of tensors ready to be batched.
    """
    # 定义元素类型的常量
    TYPE_PAD = 0
    TYPE_PAGE_CTRL = 1
    TYPE_PANEL = 2
    TYPE_CHARACTER = 3
    TYPE_DIALOG = 4

    element_types = []
    element_indices = []
    
    # --- 1. [PAGE_CTRL] Token ---
    element_types.append(TYPE_PAGE_CTRL)
    element_indices.append(0) # 全局Token的索引为0

    # --- 2. [PANEL_{i}] Tokens ---
    panels = annotation.get('frames', [])
    for i, panel in enumerate(panels):
        element_types.append(TYPE_PANEL)
        element_indices.append(i)

    # --- 3. [CHARACTER_{k}] Tokens ---
    all_characters = [char for panel in panels for char in panel.get('characters', [])]
    for k, char in enumerate(all_characters):
        element_types.append(TYPE_CHARACTER)
        element_indices.append(k)

    # --- 4. [DIALOG_{j}] Tokens ---
    all_dialogs = [dialog for panel in panels for dialog in panel.get('dialogs', [])]
    for j, dialog in enumerate(all_dialogs):
        element_types.append(TYPE_DIALOG)
        element_indices.append(j)

    # --- 提取其他信息 ---
    # 风格向量
    style_params = annotation['style_parameters']
    style_vector = torch.tensor([
        style_params['layout_density'],
        style_params['alignment_score'],
        style_params['shape_instability'],
        style_params['breakout_intensity']
    ], dtype=torch.float32)

    return {
        "element_types": torch.tensor(element_types, dtype=torch.long),
        "element_indices": torch.tensor(element_indices, dtype=torch.long),
        "style_vector": style_vector,
    }


# ==============================================================================
# 步骤 2: Input Embedding 模块
# 这个PyTorch模块负责将Token化的张量转换为最终的嵌入向量。
# ==============================================================================

class InputEmbeddings(nn.Module):
    def __init__(self, d_model=256, max_elements=50, text_embed_dim=768):
        super().__init__()
        self.d_model = d_model

        # --- 1. 定义各类嵌入层 ---
        # 元素类型嵌入: 0=PAD, 1=PAGE_CTRL, 2=PANEL, 3=CHARACTER, 4=DIALOG
        self.element_type_embedding = nn.Embedding(5, d_model)
        
        # 元素索引嵌入 (用于区分 PANEL_1, PANEL_2, etc.)
        self.element_idx_embedding = nn.Embedding(max_elements, d_model)

        # --- 2. 定义各类映射层 (用于连续/高维向量) ---
        # 风格向量 (4维) -> d_model
        self.style_mlp = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 文本嵌入 (e.g., 768维) -> d_model
        self.text_embed_projection = nn.Linear(text_embed_dim, d_model)
        
        # --- 3. 最终的序列位置编码 ---
        # 这一层学习Token在整个序列中的绝对位置信息
        self.final_positional_encoding = nn.Embedding(max_elements * 4, d_model) # 假设最大序列长度

    def forward(self, batch):
        """
        Args:
            batch (dict): A dictionary of batched tensors from the collate_fn.
        
        Returns:
            torch.Tensor: The final input embeddings for the Transformer.
            torch.Tensor: The padding mask.
        """
        element_types = batch['element_types']
        element_indices = batch['element_indices']
        B, SeqLen = element_types.shape

        # --- a. 基础嵌入 (类型 + 自身索引) ---
        # 所有Token都有这两个基础信息
        base_embed = self.element_type_embedding(element_types) + \
                     self.element_idx_embedding(element_indices)

        # --- b. 注入特定信息 ---
        # 创建一个掩码来定位PAGE_CTRL token
        page_ctrl_mask = (element_types == 1).unsqueeze(-1) # Shape: (B, SeqLen, 1)
        
        # 计算风格嵌入
        style_embed = self.style_mlp(batch['style_vector']).unsqueeze(1) # (B, 1, d_model)

        # 将这些信息加到PAGE_CTRL token的嵌入上
        # 使用广播机制，只有掩码为True的位置会被加上
        base_embed = base_embed + style_embed  * page_ctrl_mask
        
        # --- c. 加入最终的全局序列位置编码 ---
        position_ids = torch.arange(SeqLen, device=base_embed.device).unsqueeze(0).expand(B, SeqLen)
        final_embed = base_embed + self.final_positional_encoding(position_ids)
        
        # --- d. 生成Padding Mask ---
        # Transformer需要知道哪些是padding token以便忽略它们
        padding_mask = (element_types == 0) # 0 is PAD
        
        return final_embed, padding_mask

# ==============================================================================
# 步骤 3: Collate Function
# 用于DataLoader，将多个样本打包成一个批次，并进行padding。
# ==============================================================================

def custom_collate_fn(batch_list):
    """
    Pads sequences to the max length in the batch and stacks other tensors.
    """
    collated = {}
    # 获取批次中第一个样本的所有键
    keys = batch_list[0].keys()

    for key in keys:
        items = [d[key] for d in batch_list]
        if key in ["element_types", "element_indices"]:
            # 对序列进行padding
            collated[key] = pad_sequence(items, batch_first=True, padding_value=0) # 0 is PAD
        elif key in ["style_vector", "page_caption_embedding"]:
            # 直接堆叠
            collated[key] = torch.stack(items, dim=0)
        else:
            # 对于panel_caption_embeddings等，暂时先不处理，放入列表中
            collated[key] = items
            
    return collated

if __name__ == '__main__':
    # --- 模拟环境 ---
    # 模拟一个文本编码器
    def mock_text_encoder(text):
        return torch.randn(768)

    ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_style.json"
    # --- 数据处理 ---
    processed_data = process_annotation_to_tensors(annotation, mock_text_encoder)
    
    # --- 打包成批次 ---
    batch = custom_collate_fn([processed_data])
    
    print("--- Batch Content ---")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"Key: {key}, Shape: {value.shape}")
        else:
            print(f"Key: {key}, Type: list")
    
    # --- 模型前向传播 ---
    d_model = 256
    embedding_module = InputEmbeddings(d_model=d_model)
    
    final_embeddings, padding_mask = embedding_module(batch)
    
    print("\n--- Model Output ---")
    print(f"Final Embeddings Shape: {final_embeddings.shape}")
    print(f"Padding Mask Shape: {padding_mask.shape}")
    print(f"Padding Mask Content:\n{padding_mask}")

    # 验证输出维度是否正确
    assert final_embeddings.shape[0] == 2 # Batch size
    assert final_embeddings.shape[2] == d_model # Embedding dimension
