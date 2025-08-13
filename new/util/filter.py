
import json

def filter_annotations(input_file, output_file):
    # 读取原始JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 过滤数据
    filtered_data = [
        item for item in annotations 
        if item["image_path"].startswith("assassination-classroom/")
    ]
    
    # 保存过滤后的数据到新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"过滤完成，共保留 {len(filtered_data)} 条数据")

# 使用示例
input_json = "DiffSensei-main/checkpoints/mangazero/annotations.json"  # 输入文件路径
output_json = "DiffSensei-main/checkpoints/mangazero/filtered_annotations.json"  # 输出文件路径
filter_annotations(input_json, output_json)