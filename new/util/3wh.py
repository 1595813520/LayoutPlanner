import json
from PIL import Image
import os
from typing import List, Dict

def insert_image_dimensions(annotation: List[Dict], img_root: str) -> List[Dict]:
    """
    为标注数据中的每个条目添加width和height属性，插入在image_path下方
    图片路径由 img_root + image_path 拼接而成
    
    参数:
        annotation: 原始标注数据列表
        img_root: 图片根目录路径
    返回:
        添加宽高属性后的标注数据
    """
    processed = []
    for item in annotation:
        # 复制原始条目避免修改源数据
        item_copy = item.copy()
        
        # 获取图片路径并拼接完整路径
        image_path = item_copy.get('image_path')  # 从item中获取相对路径
        if image_path:
            # 拼接完整图片路径：img_root + image_path
            full_image_path = os.path.join(img_root, image_path)
            abs_image_path = os.path.abspath(full_image_path)  # 转为绝对路径便于调试
            
            try:
                with Image.open(abs_image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"警告: 无法读取图片 {abs_image_path}, 错误: {e}")
                width, height = 0, 0  # 读取失败时使用默认值
            
            # 将字典转为有序列表以保持插入位置（Python 3.7+ 字典有序）
            items_list = list(item_copy.items())
            for i, (key, _) in enumerate(items_list):
                if key == 'image_path':
                    # 在image_path后插入width和height（保持顺序）
                    items_list.insert(i + 1, ('width', width))
                    items_list.insert(i + 2, ('height', height))
                    break
            item_copy = dict(items_list)
        
        processed.append(item_copy)
    return processed

def remove_unwanted_attributes(annotation: List[Dict]) -> List[Dict]:
    """
    移除标注数据中所有的mask_rle和shape_features属性
    
    参数:
        annotation: 原始标注数据列表
    返回:
        移除指定属性后的标注数据
    """
    processed = []
    for item in annotation:
        item_copy = item.copy()
        
        # 处理frames
        if 'frames' in item_copy:
            processed_frames = []
            for frame in item_copy['frames']:
                frame_copy = frame.copy()
                
                # 移除frame中的mask_rle和shape_features
                if 'mask_rle' in frame_copy:
                    del frame_copy['mask_rle']
                if 'shape_features' in frame_copy:
                    del frame_copy['shape_features']
                
                # 处理dialogs
                if 'dialogs' in frame_copy:
                    processed_dialogs = []
                    for dialog in frame_copy['dialogs']:
                        dialog_copy = dialog.copy()
                        # 移除dialog中的指定属性
                        if 'mask_rle' in dialog_copy:
                            del dialog_copy['mask_rle']
                        if 'shape_features' in dialog_copy:
                            del dialog_copy['shape_features']
                        processed_dialogs.append(dialog_copy)
                    frame_copy['dialogs'] = processed_dialogs
                
                processed_frames.append(frame_copy)
            item_copy['frames'] = processed_frames
        
        processed.append(item_copy)
    return processed

# 执行示例
if __name__ == "__main__":
    # 配置路径
    ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_breakout.json"
    OUT_ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_breakout_new.json"
    IMG_ROOT = "/data/DiffSensei-main/checkpoints/mangazero/images"  # 图片根目录
    
    # 读取原始标注
    with open(ANN_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 执行处理流程：先插入宽高，再移除属性
    data_with_dimensions = insert_image_dimensions(data, IMG_ROOT)  # 传入img_root参数
    final_data = remove_unwanted_attributes(data_with_dimensions)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUT_ANN_FILE), exist_ok=True)
    
    # 保存结果
    with open(OUT_ANN_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果保存至: {OUT_ANN_FILE}")