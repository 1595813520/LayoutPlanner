# train.py
import os
import time
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.datasets import MangaLayoutDataset
from utils.collate import collate_fn
from models.layout_planner.planner import LayoutPlanner
from utils.losses import LayoutCompositeLoss

def parse_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def apply_style_cfg_dropout(style_vectors, p):
    """
    style_vectors: Tensor (B,4)
    p: dropout prob (0..1)
    For each sample with prob p, zero out style vector (unconditional).
    """
    if p <= 0.0:
        return style_vectors
    device = style_vectors.device
    B = style_vectors.shape[0]
    mask = (torch.rand(B, device=device) >= p).float().unsqueeze(-1)  # 1 means keep, 0 means drop
    return style_vectors * mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="planner.yaml")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument("--batch_size", type=int, default=8, help="")
    args = parser.parse_args()

    cfg = parse_yaml(args.config)

    # dataset config
    ds_cfg = cfg["dataset"]
    train_cfg = ds_cfg["train"]
    model_params = cfg["model"]["parameters"]
    train_params = cfg.get("training", {})
    layout_types = cfg.get("layout_types", {})
    panel_shapes = cfg.get("panel_shapes", {})
    bubble_shapes = cfg.get("bubble_shapes", {})
    
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & Loader
    dataset = MangaLayoutDataset(
        ann_source=train_cfg["data_path"],
        image_dir=train_cfg.get("image_dir", None),
        cfg={
            "max_elements": ds_cfg.get("max_elements", 100),
             "max_panels": ds_cfg["parameters"].get("max_panels", 18),
             "max_dialogs":  ds_cfg["parameters"].get("max_dialogs", 12),
             "max_chars":    ds_cfg["parameters"].get("max_characters", 12),
             "layout_types": layout_types,
             "panel_shapes": panel_shapes
             }
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=train_cfg.get("shuffle", True),
        num_workers=ds_cfg["parameters"]["loader_num_workers"],
        # collate_fn=lambda bl: collate_fn(bl, {
        #     "max_elements": ds_cfg.get("max_elements", 100),
        #     "max_panels": dataset.cfg["max_panels"],
        #     "max_dialogs": dataset.cfg["max_dialogs"],
        #     "max_chars": dataset.cfg["max_chars"],
        # }),
        collate_fn=lambda batch: collate_fn(batch, cfg),
        pin_memory=True,
    )

    # model

    planner = LayoutPlanner(
        encoder_cfg={
            **model_params, 
            "layout_types": layout_types
        },
        heads_cfg={
            "num_panel_classes": len(panel_shapes),
            "num_dialog_shapes": len(bubble_shapes),  # 不预测 bubble shape
        }
    ).to(device)

    # loss & optimizer
    lambda_style = train_params.get("lambda_style", 0.1)
    criterion = LayoutCompositeLoss(lambda_style=lambda_style).to(device)
    
    lr = float(train_params.get("lr", 1e-4))  # 转换为浮点型
    weight_decay = float(train_params.get("weight_decay", 1e-2))  # 转换为浮点型
    optimizer = torch.optim.AdamW(planner.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(planner.parameters(), lr=train_params.get("lr", 1e-4), weight_decay=train_params.get("weight_decay", 1e-2))

    # CFG dropout param
    style_dropout_p = float(train_params.get("style_cfg_dropout", 0.0))

    save_dir = train_params.get("save_dir", "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(train_params.get("epochs", 20))
    clip_grad = float(train_params.get("clip_grad_norm", 1.0))

    best_loss = float("inf")
    global_step = 0
    for epoch in range(1, epochs+1):
        planner.train()
        running = 0.0
        t0 = time.time()
        for it, batch in enumerate(loader):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else
                         {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v)
                     for k, v in batch.items()}

            # Apply CFG-style dropout to style_vector
            batch["style_vector"] = apply_style_cfg_dropout(batch["style_vector"], style_dropout_p)

            outputs = planner(batch)  # list of per-sample outputs

            loss, logs = criterion(outputs, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(planner.parameters(), clip_grad)
            optimizer.step()

            running += float(loss.item())
            global_step += 1

            if (it + 1) % 50 == 0:
                avg = running / (it + 1)
                print(f"[Epoch {epoch} | iter {it+1}/{len(loader)}] avg_loss={avg:.4f} geom={logs['geom_loss']:.4f} style={logs['style_loss']:.4f}")

        epoch_loss = running / max(1, len(loader))
        print(f"Epoch {epoch} done in {time.time()-t0:.1f}s | loss={epoch_loss:.4f}")

        # save
        ckpt = os.path.join(save_dir, f"planner_epoch{epoch}.pt")
        torch.save({"model": planner.state_dict(), "epoch": epoch}, ckpt)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"model": planner.state_dict(), "epoch": epoch}, os.path.join(save_dir, "planner_best.pt"))

    print("Training finished.")
    print(f"Best model saved to {os.path.join(save_dir, 'planner_best.pt')}")
    
if __name__ == "__main__":
    main()
