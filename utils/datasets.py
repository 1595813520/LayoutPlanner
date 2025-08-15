# utils/datasets.py
import os
import json
from torch.utils.data import Dataset

class MangaLayoutDataset(Dataset):
    def __init__(self, ann_source, image_dir=None, cfg=None):
        self.cfg = cfg
        self.samples = []
        if os.path.isdir(ann_source):
            files = [os.path.join(ann_source, f) for f in os.listdir(ann_source) if f.endswith(".json")]
            files.sort()
            for p in files:
                with open(p, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                self.samples.append(ann)
        else:
            with open(ann_source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.samples = data
            elif isinstance(data, dict) and "frames" in data:
                self.samples = [data]
            elif isinstance(data, dict) and "annotations" in data:
                self.samples = data["annotations"]
            else:
                raise ValueError(f"Unsupported JSON format: {ann_source}")

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No annotations found from {ann_source}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 直接返回单条原始 annotation dict
        return self.samples[idx]