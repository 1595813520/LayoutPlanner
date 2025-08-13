import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_breakout.json"
OUT_IMG = "/data/DiffSensei-main/checkpoints/mangazero/breakout_elements_hist_mask.png"

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# 统计所有breakout_area的分布，或者按区间分bins
all_breakout_ratios = []
for page in tqdm(annotations):
    for frame in page["frames"]:
        be = frame.get("breakout_elements", [])
        for elem in be:
            ratio = elem.get("breakout_area", 0)
            all_breakout_ratios.append(ratio)

# 分成区间统计柱状数据
bins = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # 可以根据需要调整区间
hist, bin_edges = np.histogram(all_breakout_ratios, bins=bins)

plt.figure(figsize=(8,5))
plt.bar(range(len(hist)), hist, width=0.7, align='center', color='#1f77b4')
plt.xticks(range(len(hist)), [
    "0-0.02", "0.02-0.05", "0.05-0.1", "0.1-0.15", "0.15-0.2", "0.2-0.25", "0.25-0.3"
])
plt.xlabel('Breakout Ratio distribution')
plt.ylabel('number of Elements')
plt.tight_layout()
plt.savefig(OUT_IMG)
plt.close()

print(f"柱状统计图已保存: {OUT_IMG}")