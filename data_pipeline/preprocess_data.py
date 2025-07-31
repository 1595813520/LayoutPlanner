import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MangaLayoutDataset(Dataset):
    """
    漫画布局数据集。
    假设数据已被预处理为.pt文件，其中包含一个字典列表。
    每个字典代表一个页面，包含几何真值和风格真值。
    """
    def __init__(self, data_path):
        # 加载预处理好的数据
        # self.data 应该是一个列表，例如:
        # [
        #   {
        #     "elements_type": tensor([...]), # 1 for panel, 2 for dialog, ...
        #     "gt_geometry": tensor([...]),   # (N, 8) for panels, (M, 4) for dialogs
        #     "gt_shapes": tensor([...]),     # 类别标签
        #     "style_vector": tensor([...]),  # 5维的风格真值 S_gt
        #     "caption_embeds": tensor([...]) # 预计算的文本嵌入
        #   }, ...
        # ]
        self.data = torch.load(data_path)
        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """
    自定义的collate_fn，用于处理变长序列，将其padding为统一长度。
    """
    collated_batch = {}
    keys = batch[0].keys()

    for key in keys:
        # 提取当前key的所有数据
        items = [d[key] for d in batch]
        
        if isinstance(items[0], torch.Tensor) and items[0].ndim > 0:
            # 对张量序列进行padding
            # batch_first=True 表示输出的shape为 [batch_size, seq_len, ...]
            collated_batch[key] = pad_sequence(items, batch_first=True, padding_value=0)
        else:
            # 对于非序列数据（如style_vector），直接堆叠
            collated_batch[key] = torch.stack(items, dim=0) if isinstance(items[0], torch.Tensor) else items

    return collated_batch