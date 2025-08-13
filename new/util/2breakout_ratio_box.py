import os
import json
from tqdm import tqdm
import math

ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_4pts.json"
OUT_ANN_FILE = "/data/DiffSensei-main/checkpoints/mangazero/f_annotations_with_breakout.json"

with open(ANN_FILE, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

def is_point_inside_polygon(x, y, poly):
    """Check if point (x,y) is inside polygon defined by list of [x,y] points"""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        if ((poly[i][1] > y) != (poly[j][1] > y) and
                x < (poly[j][0] - poly[i][0]) * (y - poly[i][1]) / 
                (poly[j][1] - poly[i][1] + 1e-10) + poly[i][0]):
            inside = not inside
        j = i
    return inside

def compute_breakout_ratio(box_char, four_points):
    """Calculate breakout ratio using character bbox and panel classification points"""
    xA, yA, xB, yB = box_char
    
    # Create a grid of points within character bbox
    grid_size = 20  # Number of points per dimension
    x_coords = [xA + i * (xB - xA) / grid_size for i in range(grid_size + 1)]
    y_coords = [yA + i * (yB - yA) / grid_size for i in range(grid_size + 1)]
    
    # Count points inside the panel polygon
    inside_points = 0
    total_points = (grid_size + 1) * (grid_size + 1)
    
    for x in x_coords:
        for y in y_coords:
            if is_point_inside_polygon(x, y, four_points):
                inside_points += 1
    
    # Calculate ratio of points outside the panel
    char_area = max(1, (xB - xA) * (yB - yA))
    ratio = 1.0 - (inside_points / total_points)
    
    return max(0.0, min(1.0, ratio))

for page in tqdm(annotations):
    for fi, frame in enumerate(page["frames"]):
        # Get panel classification points
        four_points = frame.get("four_points", None)
        if four_points is None or len(four_points) < 3:
            continue
        breakout_elements = []
        if "characters" in frame:
            for char in frame["characters"]:
                char_box = char.get("bbox", None)
                # cid = char.get("id", None)
                # if char_box is None or cid is None:
                if char_box is None:
                    continue
                ratio = compute_breakout_ratio(char_box, four_points)
                char["breakout_ratio"] = ratio
                
        if "dialogs" in frame:
            for dialog in frame["dialogs"]:
                dialog_box = dialog.get("rect_box", None)
                if dialog_box is None:
                    continue
                ratio = compute_breakout_ratio(dialog_box, four_points)
                dialog["breakout_ratio"] = ratio

with open(OUT_ANN_FILE, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)