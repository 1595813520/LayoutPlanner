def preprocess_manga_page(image_path):
    # 1. 统一检测
    results = yolo_model(image_path)
    
    # 2. 形状分析
    panels = []
    for detection in results:
        if detection['class'].startswith('panel'):
            features = analyze_panel_shape(detection['mask'])
            shape_type = classify_panel_shape(features)
            panels.append({
                'bbox': detection['bbox'],
                'mask': detection['mask'],
                'shape_type': shape_type,
                'features': features
            })
    
    # 3. 破格分析
    breakout_info = analyze_breakout(results)
    
    # 4. 多样性指数计算
    diversity = calculate_diversity_index(panels, breakout_info)
    
    return {
        'panels': panels,
        'other_elements': [...],
        'diversity': diversity
    }
    
def generate_training_data(manga_dataset):
    training_pairs = []
    
    for page in manga_dataset:
        annotation = preprocess_manga_page(page)
        
        # Layout planning数据
        layout_input = {
            'diversity_target': annotation['diversity'],
            'content_desc': extract_content_description(annotation)
        }
        layout_output = {
            'panel_layout': annotation['panels']
        }
        
        training_pairs.append((layout_input, layout_output))
    
    return training_pairs