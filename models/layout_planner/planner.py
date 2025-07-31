import torch
import torch.nn as nn

# 假设这些模块在相应的文件中已定义
from .embeddings import InputEmbeddings
from .lfm import LayoutFusionModule
from .heads import GeometryHead, TypeHead

class LayoutPlanner(nn.Module):
    """
    漫画布局规划器主模型。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. 输入嵌入层
        self.embeddings = InputEmbeddings(
            d_model=config.d_model,
            style_vec_dim=5, # 我们的风格向量是5维
            max_elements=config.dataset.max_elements
        )

        # 2. 布局融合模块 (LFM)
        self.lfm = LayoutFusionModule(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout
        )

        # 3. 并行预测头
        # 预测panel的8个坐标 (4个点)
        self.panel_geometry_head = GeometryHead(config.d_model, 8) 
        # 预测dialog的4个坐标 (bbox)
        self.dialog_bbox_head = GeometryHead(config.d_model, 4)
        # 预测形状类型的logits
        self.shape_type_head = TypeHead(config.d_model, num_classes=10) # 假设有10种形状

    def forward(self, batch):
        """
        前向传播逻辑。
        """
        # 从batch中提取所需输入
        style_vector = batch['style_vector']
        element_types = batch['elements_type']
        caption_embeds = batch['caption_embeds']
        
        # 1. 获取输入嵌入
        # `padding_mask` 用于在LFM中忽略padding部分
        input_embeds, padding_mask = self.embeddings(style_vector, element_types, caption_embeds)
        
        # 2. 通过LFM进行全局信息融合
        # (B, SeqLen, d_model) -> (B, SeqLen, d_model)
        fused_features = self.lfm(input_embeds, src_key_padding_mask=padding_mask)
        
        # 3. 并行解码，为不同类型的元素使用不同的头
        # 创建一个字典来存储预测结果
        predicted_layout = {'panels': {}, 'dialogs': {}, 'elements': {}}

        # 根据element_types分离出不同元素的特征
        panel_mask = (element_types == 1) # 假设1代表panel
        dialog_mask = (element_types == 2) # 假设2代表dialog

        # 对panel进行预测
        panel_features = fused_features[panel_mask]
        predicted_layout['panels']['coords'] = self.panel_geometry_head(panel_features)

        # 对dialog进行预测
        dialog_features = fused_features[dialog_mask]
        predicted_layout['dialogs']['bbox'] = self.dialog_bbox_head(dialog_features)

        # 对所有元素预测形状类型
        predicted_layout['elements']['shape_logits'] = self.shape_type_head(fused_features)
        
        return predicted_layout