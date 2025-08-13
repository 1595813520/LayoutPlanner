import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils
import cv2

# ==== 通用工具 ====
def decode_mask(mask_rle, image_shape):
    mask = mask_utils.decode(mask_rle)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    assert mask.shape == image_shape[:2], f"Mask shape mismatch: mask {mask.shape}, img {image_shape}"
    return mask.astype(np.uint8)
def reorder_four_points(pts):
    # 输入四个点，返回左上-右上-右下-左下顺序，保证offset含义一致
    pts = np.array(pts)
    idx = np.argsort(pts[:, 1])
    top_two = pts[idx[:2]]
    bottom_two = pts[idx[2:]]
    top_left, top_right = sorted(top_two, key=lambda p: p[0])
    bottom_left, bottom_right = sorted(bottom_two, key=lambda p: p[0])
    return [top_left, top_right, bottom_right, bottom_left]

def is_convex_quad(pts):
    # 检查四边形凸性
    pts = np.array(pts)
    if pts.shape != (4,2): return False
    def cross(a, b, c):
        ab = b - a
        ac = c - a
        return ab[0]*ac[1] - ab[1]*ac[0]
    flag = None
    for i in range(4):
        z = cross(pts[i], pts[(i+1)%4], pts[(i+2)%4])
        if z==0: continue
        if flag is None: flag = z>0
        elif (z>0)!=flag: return False
    return True

def box_to_four_points(box):
    """
    将box（[x1, y1, x2, y2]）转换为四点格式 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    """
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def calculate_polygon_area(points):
    """
    计算多边形面积（使用鞋带公式）
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calculate_intersection_area(poly1, poly2):
    """
    计算两个四边形的交集面积（使用 OpenCV 的布尔运算）
    """
    H, W = 10000, 10000  # 假设足够大的画布
    mask1 = np.zeros((H, W), dtype=np.uint8)
    mask2 = np.zeros((H, W), dtype=np.uint8)
    
    cv2.fillPoly(mask1, [np.array(poly1, dtype=np.int32)], 1)
    cv2.fillPoly(mask2, [np.array(poly2, dtype=np.int32)], 1)
    
    intersection = cv2.bitwise_and(mask1, mask2)
    return cv2.countNonZero(intersection)

def calculate_overlap_ratio(four_points, box_points):
    """
    计算 four_points 和 box_points 的重叠比例
    """
    four_points = np.array(four_points, dtype=np.float32)
    box_points = np.array(box_points, dtype=np.float32)
    
    area1 = calculate_polygon_area(four_points)
    area2 = calculate_polygon_area(box_points)
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    intersection_area = calculate_intersection_area(four_points, box_points)
    min_area = min(area1, area2)
    
    return intersection_area / min_area if min_area > 0 else 0.0

def calculate_offsets(four_points, classification_points, shape_type):
    """
    计算 classification_points 相对于 four_points 的偏移量
    如果 shape_type 是 panel_rect，则返回全 0
    """
    if shape_type == 'panel_rect':
        return [0, 0, 0, 0, 0, 0, 0, 0]
    
    # 确保输入是 Python 列表
    four_points = [[int(x), int(y)] for x, y in four_points]
    classification_points = [[int(x), int(y)] for x, y in classification_points]
    
    if len(classification_points) < 4:
        return [0, 0, 0, 0, 0, 0, 0, 0]
    
    offsets = []
    for i in range(4):
        dx = classification_points[i][0] - four_points[i][0]
        dy = classification_points[i][1] - four_points[i][1]
        offsets.extend([dx, dy])
    
    return offsets

# ==== Panel shape分析 - 优化版本 ====
def polygon_angles(pts):
    """计算多边形各顶点的内角"""
    angles = []
    n = len(pts)
    for i in range(n):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        v1 = np.array(p0) - np.array(p1)
        v2 = np.array(p2) - np.array(p1)
        ang = np.arccos(
            np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0)
        )
        angles.append(np.degrees(ang))
    return angles

def is_parallel(v1, v2, tol=19):
    """判断两向量是否平行"""
    angle = np.degrees(
        np.arccos(
            np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0)
        )
    )
    return (abs(angle) < tol or abs(angle - 180) < tol)

def points_close(p1, p2, thres=16):
    """判断两点是否接近"""
    return np.linalg.norm(np.array(p1) - np.array(p2)) < thres

def find_close_points(pts, thres=16):
    """找出所有距离过近的点对"""
    close_pairs = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if points_close(pts[i], pts[j], thres):
                close_pairs.append((i, j))
    return close_pairs

def get_stable_geometric_target(contour):
    """
    获取稳定的几何目标 - 新流程核心函数
    使用minAreaRect获得稳定的四边形作为所有Panel的几何训练目标
    """
    rect = cv2.minAreaRect(contour)
    four_points = cv2.boxPoints(rect)
    four_points = np.int0(four_points)
    return four_points.tolist() 

def get_stable_shape_classification(contour):
    """
    获取稳定的形状分类 - 新流程核心函数
    通过凸包 + 多边形拟合的方式获得更稳定的形状分类
    """
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    pts = approx.reshape(-1, 2)
    return pts.tolist()

def classify_panel_shape_by_pts(pts):
    """
    根据拟合点分类Panel形状 - 简化版本
    只保留四种分类：三角形、矩形、梯形、不规则四边形
    """
    n = len(pts)
    pts = np.array(pts)
    
    if n == 3:
        return 'panel_triangle'
    
    close_pairs = find_close_points(pts, thres=16)
    if close_pairs:
        if n == 4 and len(close_pairs) >= 1:
            return 'panel_triangle'
    
    if n == 4:
        v = [pts[(i+1)%4] - pts[i] for i in range(4)]
        lens = [np.linalg.norm(vec) for vec in v]
        angles = polygon_angles(pts)
        
        is_all_90_strict = all(abs(a - 90) < 4 for a in angles)
        is_opp_parallel_strict = (is_parallel(v[0], v[2], tol=4) and 
                               is_parallel(v[1], v[3], tol=4))
        is_opposite_equal_strict = (abs(lens[0] - lens[2])/max(lens[0], lens[2]) < 0.1 and 
                                 abs(lens[1] - lens[3])/max(lens[1], lens[3]) < 0.1)
        
        parallel_0_2 = is_parallel(v[0], v[2], tol=4)
        parallel_1_3 = is_parallel(v[1], v[3], tol=4)
        
        if parallel_0_2 != parallel_1_3:
            return 'panel_trapezoid'
        
        if is_all_90_strict and is_opp_parallel_strict and is_opposite_equal_strict:
            return 'panel_rect'
        
        return 'panel_irregular_quad'
    
    return 'panel_irregular_quad'

def analyze_panel_shapes_new(annotations, IMG_ROOT):
    shape_stats = {}
    for page in tqdm(annotations, desc="分析Panel形状"):
        img_fp = os.path.join(IMG_ROOT, page["image_path"])
        try:
            image = np.array(Image.open(img_fp).convert("RGB"))
            H, W = image.shape[:2]
        except:
            H, W = page["frames"][0].get("mask_rle", {}).get("size", (None, None))
            if H is None:
                continue
        
        for i, frame in enumerate(page["frames"]):
            # 默认值
            # assign_box = lambda: [
            #     box_to_four_points(frame['box']) if 'box' in frame and frame['box'] is not None else [[0,0]]*4,
            #     'panel_rect',
            #     [0,0,0,0,0,0,0,0]
            # ]
            def assign_box():
                # 从原始box生成四点坐标（确保非零）
                four_points = box_to_four_points(frame['bbox'])
                shape_type = 'panel_rect'
                offsets = [0, 0, 0, 0, 0, 0, 0, 0]
                return four_points, shape_type, offsets
            
            if 'mask_rle' in frame and frame['mask_rle'] is not None:
                try:
                    mask = decode_mask(frame['mask_rle'], (H, W))
                    # 用凸包而非原始contour
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        four_points, shape_type, offsets = assign_box()
                        frame['shape_type'], frame['four_points'], frame['classification_points'], frame['offsets'] = shape_type, four_points, four_points, offsets
                        continue
                    biggest = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(biggest)
                    epsilon = 0.02*cv2.arcLength(hull,True)
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                    pts = approx.reshape(-1,2)

                    ## 四个点拟合 (最小外接矩形)
                    rect = cv2.minAreaRect(hull)
                    quad = cv2.boxPoints(rect)
                    quad = np.array(quad, dtype=np.float32)
                    quad = reorder_four_points(quad)
                    pts = reorder_four_points(pts) if len(pts)==4 else pts.tolist()
                    
                    # 检查内凹与离谱情况
                    is_panel = True
                    if len(pts)!=4 or not is_convex_quad(pts):
                        is_panel = False # 非凸、非四点
                    # 和quad重叠面积低也判fail
                    box_points = box_to_four_points(frame['box']) if 'box' in frame and frame['box'] is not None else None
                    overlap = calculate_overlap_ratio(quad, pts) if len(pts)==4 else 0
                    if overlap < 0.4  or not is_panel:
                        # 错误面板，回退
                        four_points, shape_type, offsets = assign_box()
                        frame['shape_type'], frame['four_points'], frame['classification_points'], frame['offsets'] = shape_type, four_points, four_points, offsets
                        continue

                    # 是否是矩形逻辑：角度接近90、对边平行且等长
                    angles = polygon_angles(np.array(pts))
                    v = [np.array(pts[(j+1)%4]) - np.array(pts[j]) for j in range(4)]
                    lens = [np.linalg.norm(vi) for vi in v]
                    angle_ok = all(abs(a-90)<7 for a in angles)
                    opp_parallel = is_parallel(v[0], v[2], tol=7) and is_parallel(v[1], v[3], tol=7)
                    opp_length = abs(lens[0]-lens[2])/max(lens[0], lens[2])<0.12 and abs(lens[1]-lens[3])/max(lens[1], lens[3])<0.12

                    if angle_ok and opp_parallel and opp_length:
                        shape_type = 'panel_rect'
                        offsets = [0,0,0,0,0,0,0,0]
                        frame['classification_points'] = quad
                        frame['four_points'] = quad
                    else:
                        # 判断梯形: 只一组对边平行
                        parallel_0_2 = is_parallel(v[0], v[2], tol=7)
                        parallel_1_3 = is_parallel(v[1], v[3], tol=7)
                        if (parallel_0_2 != parallel_1_3) and is_convex_quad(pts):
                            shape_type = 'panel_trapezoid'
                            # offsets: 只允许一组有大偏移——找主偏移点
                            off = []
                            for k in range(4):
                                d = np.array(pts[k])-np.array(quad[k])
                                off.extend([int(round(d[0])),int(round(d[1]))])
                            # 若三点或全点偏移大，说明拟合错误！
                            nonzero = np.count_nonzero(np.array(off))
                            if nonzero>4: # 超过两点
                                # 拟合失败，退回box
                                four_points, shape_type, offsets = assign_box()
                                frame['shape_type'], frame['four_points'], frame['classification_points'], frame['offsets'] = shape_type, four_points, four_points, offsets
                                continue
                            offsets = off
                            frame['classification_points'] = pts
                            frame['four_points'] = quad
                        else:
                            # 其他情况，非正常凸四边形
                            four_points, shape_type, offsets = assign_box()
                            frame['shape_type'], frame['four_points'], frame['classification_points'], frame['offsets'] = shape_type, four_points, four_points, offsets
                            continue

                    frame['shape_type'] = shape_type
                    frame['four_points'] = [list(map(int, p)) for p in frame['four_points']]
                    frame['classification_points'] = [list(map(int, p)) for p in frame['classification_points']]
                    frame['offsets'] = [int(o) for o in offsets]
                    shape_stats[shape_type] = shape_stats.get(shape_type, 0)+1
                
                except Exception as e:
                    print(f"[Panel形状分析失败]: {page['image_path']} frame{i}: {e}")
                    four_points, shape_type, offsets = assign_box()
                    frame['shape_type'], frame['four_points'], frame['classification_points'], frame['offsets'] = shape_type, four_points, four_points, offsets

    print("\n=== Panel形状分类统计 ===")
    for shape_type, count in sorted(shape_stats.items()):
        print(f"{shape_type}: {count}")
    print("========================\n")

# ==== Bubble形状分析 ====
def analyze_Bubble_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area == 0 or perimeter == 0:
        return None, None
    bbox = cv2.boundingRect(contour)
    
    # 计算轴对齐最小外接矩形（不旋转）
    x, y, w, h = cv2.boundingRect(contour)  # x_min, y_min, width, height
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h
    rect_box = [x_min, y_min, x_max, y_max]  # 两点坐标格式
    
    roundness = 4 * np.pi * area / (perimeter ** 2)
    rectangularity = area / (bbox[2] * bbox[3])
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area if hull_area > 0 else 0
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corner_count = len(approx)
    compactness = perimeter ** 2 / area
    aspect_ratio = bbox[3] / bbox[2] if bbox[2] != 0 else 0
    features = {
        'roundness': roundness,
        'rectangularity': rectangularity,
        'convexity': convexity,
        'corners': corner_count,
        'compactness': compactness,
        'aspect_ratio': aspect_ratio,
        'irregularity': 1 - min(roundness, rectangularity)
    }
    # rect = cv2.minAreaRect(contour)
    # box_points = cv2.boxPoints(rect)
    # box_points = np.int0(box_points)
    return features, rect_box

def classify_bubble_shape(features):
    if features['rectangularity'] > 0.8 and 4 <= features['corners'] <= 5:
        return 'bubble_rect'
    if features['roundness'] > 0.68 and features['corners'] <= 7 and features['aspect_ratio'] <= 2.2:
        return 'bubble_oval'
    if features['corners'] >= 10 and features['irregularity'] > 0.39 and features['convexity'] < 0.93:
        return 'bubble_burst'
    if features['corners'] >= 8 and features['convexity'] > 0.90 and 0.5 < features['roundness'] < 0.72:
        return 'bubble_flower'
    return 'bubble_irregular'

def analyze_bubble_shapes(annotations, IMG_ROOT):
    for page in tqdm(annotations, desc="分析Bubble形状"):
        img_fp = os.path.join(IMG_ROOT, page["image_path"])
        try:
            image = np.array(Image.open(img_fp).convert("RGB"))
            H, W = image.shape[:2]
        except:
            H, W = page["frames"][0].get("mask_rle", {}).get("size", (None, None))
            if H is None:
                continue
        for i, frame in enumerate(page["frames"]):
            if 'dialogs' in frame:
                for j, dialog in enumerate(frame["dialogs"]):
                    if 'mask_rle' in dialog and dialog['mask_rle'] is not None:
                        try:
                            mask = decode_mask(dialog['mask_rle'], (H, W))
                            features, rect_box = analyze_Bubble_shape(mask)
                            if features:
                                bubble_type = classify_bubble_shape(features)
                                dialog['bubble_type'] = bubble_type
                                # dialog['shape_features'] = features
                                # dialog['rect_box'] = rect_box.tolist()
                                dialog['rect_box'] = rect_box
                            else:
                                if 'box' in dialog and dialog['box'] is not None:
                                    dialog['bubble_type'] = 'bubble_oval'
                                    dialog['rect_box'] = box_to_four_points(dialog['box'])
                                # else:
                                #     dialog['bubble_type'] = 'bubble_irregular'
                                #     dialog['rect_box'] = [[0, 0], [0, 0], [0, 0], [0, 0]]
                        except Exception as e:
                            print(f"[Bubble形状分析失败]: {page['image_path']} frame{i}-dialog{j}: {e}")
                            if 'box' in dialog and dialog['box'] is not None:
                                dialog['bubble_type'] = 'bubble_oval'
                                dialog['rect_box'] = box_to_four_points(dialog['box'])
                            # else:
                            #     dialog['bubble_type'] = 'bubble_oval'
                            #     dialog['rect_box'] = [[0, 0], [0, 0], [0, 0], [0, 0]]

# ==== 主程序 ====
if __name__ == "__main__":
    ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_mask_refine.json"
    IMG_ROOT = "/data/DiffSensei-main/checkpoints/mangazero/images"
    OUT_ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_4pts_1.json"

    print("加载标注文件...")
    with open(ANN_FILE, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    print("开始新的Panel形状分析流程...")
    analyze_panel_shapes_new(annotations, IMG_ROOT)
    
    print("开始Bubble形状分析...")
    analyze_bubble_shapes(annotations, IMG_ROOT)

    print("保存结果...")
    with open(OUT_ANN_FILE, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"分析完成！结果已保存到: {OUT_ANN_FILE}")