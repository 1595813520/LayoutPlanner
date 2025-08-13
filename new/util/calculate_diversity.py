import os
import json
import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

IN_JSON = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_breakout.json"
OUT_JSON = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_diversity.json"

def calculate_diversity_index(page_annotation):
    panels = page_annotation.get('frames', [])
    n_panels = len(panels)
    if n_panels == 0:
        return {
            'psd': 0,
            'ebd': 0,
            'lcd': 0,
            'sid': 0,
            'total': 0,
            'components': {}
        }
    # 1. Panel形状多样性 (PSD)
    shape_types = [p.get('shape_type', 'panel_irregular') for p in panels]
    shape_counts = Counter(shape_types)
    shape_probs = np.array(list(shape_counts.values())) / n_panels
    shape_entropy = -np.sum(shape_probs * np.log2(shape_probs + 1e-10))
    max_entropy = np.log2(min(n_panels, 5))  # 最多5种基本形状
    psd = shape_entropy / max_entropy if max_entropy > 0 else 0

    # 2. 破格多样性 (EBD)
    # 若panel中无breakout_ratio则取0，或者所有breakout_elements的均值
    breakout_avgs = []
    for p in panels:
        ratio = 0.0
        if 'breakout_elements' in p and p['breakout_elements']:
            ratio = np.mean([e.get('breakout_area', 0.0) for e in p['breakout_elements']])
        breakout_avgs.append(ratio)
    ebd = np.mean(breakout_avgs) if breakout_avgs else 0.0

    # 3. 布局复杂度 (LCD)
    centers = []
    for panel in panels:
        bbox = panel.get('bbox', None)
        if bbox:
            centers.append([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    if len(centers) > 1:
        centers = np.array(centers)
        # 直接用欧氏距离矩阵
        dist_mat = distance_matrix(centers, centers)
        mst_mat = minimum_spanning_tree(dist_mat)
        mst_length = float(mst_mat.sum())
        # 获取页面宽高
        page_w = page_h = None
        if 'meta' in page_annotation:
            page_w = page_annotation['meta'].get('width', None)
            page_h = page_annotation['meta'].get('height', None)
        else:
            # 自动推理页面最大宽高
            x_vals = [max(p['bbox'][0], p['bbox'][2]) for p in panels]
            y_vals = [max(p['bbox'][1], p['bbox'][3]) for p in panels]
            page_w, page_h = max(x_vals), max(y_vals)
        page_diag = np.sqrt(page_w ** 2 + page_h ** 2) if page_w and page_h else 1.0
        lcd = min(mst_length / page_diag, 1.0)
    else:
        mst_length = 0.0
        lcd = 0.0

    # 4. 形状不规则度 (SID)
    irregularities = [p.get('shape_features', {}).get('irregularity', 0) for p in panels]
    sid = np.mean(irregularities) if irregularities else 0.0

    # 5. 综合指数
    weights = {'psd': 0.3, 'ebd': 0.3, 'lcd': 0.2, 'sid': 0.2}
    total_diversity = (weights['psd'] * psd +
                      weights['ebd'] * ebd +
                      weights['lcd'] * lcd +
                      weights['sid'] * sid)

    return {
        'psd': psd,        # Panel Shape Diversity
        'ebd': ebd,        # Element Breakout Diversity  
        'lcd': lcd,        # Layout Complexity Diversity
        'sid': sid,        # Shape Irregularity Diversity
        'total': total_diversity,
        'components': {
            'shape_entropy': float(shape_entropy),
            'mst_length': float(mst_length if len(centers) > 1 else 0),
            'avg_irregularity': float(sid)
        }
    }

# main process
with open(IN_JSON, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

for page in tqdm(annotations):
    diversity = calculate_diversity_index(page)
    page['diversity_index'] = diversity

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)