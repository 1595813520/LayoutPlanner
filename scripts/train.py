import argparse
from venv import logger
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.distributed as dist
from omegaconf import OmegaConf
from models.layout_planner.embeddings import InputEmbeddings, process_annotation_to_tensors, custom_collate_fn
from models.layout_planner.planner import LayoutPlanner
from utils.losses import LayoutCompositeLoss


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='./layout-generator/configs/planner.yaml')
    known_args, unknown_args = parser.parse_known_args()

    known_args = OmegaConf.create(known_args.__dict__)
    cfg = OmegaConf.merge(OmegaConf.load(known_args.config_file), known_args)
    if unknown_args:
        unknown_args = OmegaConf.from_dotlist(unknown_args)
        cfg = OmegaConf.merge(cfg, unknown_args)
    print(OmegaConf.to_yaml(cfg))
    logger.log("creating model...")
     
    # 1. 初始化模型
    print("Initializing model...")
    model = LayoutPlanner(cfg.model)
    model.to(cfg.device)

    # 2. 准备数据
    print("Loading data...")
    # train_dataset = MangaLayoutDataset(cfg.dataset.train_data_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    # val_loader同理
    
    # --- 数据处理 ---
    processed_data = process_annotation_to_tensors(annotation)
    
    # --- 打包成批次 ---
    batch = custom_collate_fn([processed_data])
    
    embedding_module = InputEmbeddings(d_model=d_model)
    
    final_embeddings, padding_mask = embedding_module(batch)

    # 3. 初始化优化器和损失函数
    print("Setting up optimizer and loss...")
    optimizer = AdamW(
        model.parameters(), 
        lr=cfg.training.optimizer.lr, 
        weight_decay=cfg.training.optimizer.weight_decay
    )
    criterion = LayoutCompositeLoss(lambda_style=cfg.training.lambda_style)

    # 4. 开始训练循环
    print("Starting training...")
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_epoch_loss = 0
        
        for i, batch in enumerate(train_loader):
            # 将数据移动到GPU
            batch = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 前向传播
            predicted_layout = model(batch)

            # 计算损失
            loss, loss_dict = criterion(predicted_layout, batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
            optimizer.step()
            
            total_epoch_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{cfg.training.num_epochs}], Step [{i}/{len(train_loader)}], Losses: {loss_dict}")

        print(f"Epoch {epoch+1} average loss: {total_epoch_loss / len(train_loader)}")

        # 在每个epoch后进行验证...
        # 保存模型...

if __name__ == "__main__":
    main()