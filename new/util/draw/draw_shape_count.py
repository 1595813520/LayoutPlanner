import json
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

# ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_shape.json"
ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_shape_refine.json"
OUT_DIR = "/data/DiffSensei-main/checkpoints/mangazero/"
PANEL_PLOT = OUT_DIR + "panel_shape_type_hist_refine.png"
BUBBLE_PLOT = OUT_DIR + "bubble_type_hist.png"

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

panel_counter = Counter()
bubble_counter = Counter()

for page in tqdm(annotations):
    for frame in page["frames"]:
        if "shape_type" in frame:
            panel_counter[frame["shape_type"]] += 1
        if "dialogs" in frame:
            for dialog in frame["dialogs"]:
                if "bubble_type" in dialog:
                    bubble_counter[dialog["bubble_type"]] += 1

# 画 panel_shape_type 柱状图
panel_types = list(panel_counter.keys())
panel_counts = [panel_counter[k] for k in panel_types]

plt.figure(figsize=(10, 6))
bars = plt.bar(panel_types, panel_counts, color='#1f77b4', alpha=0.8)
plt.xlabel('Panel Shape Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Panel Shape Type Distribution', fontsize=14)

# 添加数量标注
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(PANEL_PLOT, dpi=300, bbox_inches='tight')
plt.close()
print("已保存panel shape type分布图:", PANEL_PLOT)

# 画 bubble_type 柱状图
bubble_types = list(bubble_counter.keys())
bubble_counts = [bubble_counter[k] for k in bubble_types]

plt.figure(figsize=(10, 6))
bars = plt.bar(bubble_types, bubble_counts, color='#ff7f0e', alpha=0.8)
plt.xlabel('Bubble Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Bubble Type Distribution', fontsize=14)

# 添加数量标注
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(BUBBLE_PLOT, dpi=300, bbox_inches='tight')
plt.close()
print("已保存bubble type分布图:", BUBBLE_PLOT)