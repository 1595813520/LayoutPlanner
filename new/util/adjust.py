import cv2
import json
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

# ########### 配置路径（根据你的需求修改）###########
DEFAULT_ANNOTATION_FILE = "DiffSensei-main/checkpoints/mangazero/f_annotations_with_stable_shape.json"  # 指定标注文件
ORIGINAL_IMAGE_FOLDER = "DiffSensei-main/checkpoints/mangazero/image"  # 原图文件夹
ANNOTATED_IMAGE_FOLDER = "DiffSensei-main/checkpoints/mangazero/image_mask_with_stable_shape"  # 已标注图片文件夹（用于参考）


class InteractiveAnnotationTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Panel Annotation Editor")
        self.root.geometry("1400x800")
        
        # 数据
        self.annotations = None
        self.annotation_file = DEFAULT_ANNOTATION_FILE  # 默认使用指定标注文件
        self.current_page_idx = 0
        self.current_panel_idx = 0
        self.image = None  # 原图（用于编辑）
        self.annotated_image = None  # 已标注图片（用于参考）
        self.display_image = None
        self.scale_factor = 1.0
        
        # 拖拽状态
        self.dragging_point = -1
        self.mouse_pos = (0, 0)
        
        # 界面设置
        self.setup_ui()
        
        # OpenCV窗口回调
        self.window_name = "Panel Editor"
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件控制
        file_frame = ttk.LabelFrame(control_frame, text="文件操作")
        file_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(file_frame, text="加载标注文件", command=self.load_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="保存标注", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        
        # 导航控制
        nav_frame = ttk.LabelFrame(control_frame, text="导航")
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(nav_frame, text="上一页", command=self.prev_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="下一页", command=self.next_page).pack(side=tk.LEFT, padx=5)
        
        self.page_label = ttk.Label(nav_frame, text="页面: 0/0")
        self.page_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(nav_frame, text="上一个Panel", command=self.prev_panel).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="下一个Panel", command=self.next_panel).pack(side=tk.LEFT, padx=5)
        
        self.panel_label = ttk.Label(nav_frame, text="Panel: 0/0")
        self.panel_label.pack(side=tk.LEFT, padx=10)
        
        # Panel信息和操作
        panel_frame = ttk.LabelFrame(control_frame, text="Panel操作")
        panel_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.panel_info_label = ttk.Label(panel_frame, text="当前Panel信息: 无")
        self.panel_info_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(panel_frame, text="重置为矩形框", command=self.reset_to_bbox).pack(side=tk.RIGHT, padx=5)
        ttk.Button(panel_frame, text="开始编辑", command=self.start_editing).pack(side=tk.RIGHT, padx=5)
        
        # 帮助信息
        help_frame = ttk.LabelFrame(control_frame, text="操作说明")
        help_frame.pack(fill=tk.X)
        
        help_text = ("拖拽操作: 在OpenCV窗口中拖拽红色圆点调整四点位置 | "
                    "键盘操作: 'q'退出编辑, 's'保存当前调整, 'r'重置为矩形框, 'n'下一个Panel")
        ttk.Label(help_frame, text=help_text, wraplength=800).pack(padx=5, pady=2)
        
    def load_annotations(self):
        """加载标注文件（默认使用指定文件）"""
        # 优先使用用户选择的文件，否则用默认标注文件
        file_path = filedialog.askopenfilename(
            title="选择标注文件",
            filetypes=[("JSON files", "*.json")],
            initialfile=os.path.basename(DEFAULT_ANNOTATION_FILE),
            initialdir=os.path.dirname(DEFAULT_ANNOTATION_FILE)
        ) or DEFAULT_ANNOTATION_FILE
        
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                self.annotation_file = file_path
                self.current_page_idx = 0
                self.current_panel_idx = 0
                self.update_display()
                messagebox.showinfo("成功", f"已加载 {len(self.annotations)} 页标注（来自：{os.path.basename(file_path)}）")
            except Exception as e:
                messagebox.showerror("错误", f"加载文件失败: {e}")
        else:
            messagebox.showwarning("警告", "标注文件不存在")
    
    def save_annotations(self):
        """保存标注文件"""
        if not self.annotations:
            messagebox.showwarning("警告", "没有标注数据可保存")
            return
            
        try:
            # 备份原文件
            backup_file = self.annotation_file + ".backup"
            import shutil
            shutil.copy2(self.annotation_file, backup_file)
            
            # 保存新文件
            with open(self.annotation_file, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", "标注已保存！备份文件已创建")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")
    
    def update_display(self):
        """更新显示信息"""
        if not self.annotations:
            return
            
        # 更新页面信息
        total_pages = len(self.annotations)
        self.page_label.config(text=f"页面: {self.current_page_idx + 1}/{total_pages}")
        
        # 更新Panel信息
        current_page = self.annotations[self.current_page_idx]
        total_panels = len(current_page.get("frames", []))
        self.panel_label.config(text=f"Panel: {self.current_panel_idx + 1}/{total_panels}")
        
        # 更新Panel详细信息
        if total_panels > 0 and self.current_panel_idx < total_panels:
            current_panel = current_page["frames"][self.current_panel_idx]
            shape_type = current_panel.get("shape_type", "Unknown")
            has_class_points = "classification_points" in current_panel  # 检查是否有classification_points
            info = f"类型: {shape_type}, 分类点: {'有' if has_class_points else '无'}"
            self.panel_info_label.config(text=f"当前Panel: {info}")
        else:
            self.panel_info_label.config(text="当前Panel: 无")
    
    def prev_page(self):
        if self.annotations and self.current_page_idx > 0:
            self.current_page_idx -= 1
            self.current_panel_idx = 0
            self.update_display()
    
    def next_page(self):
        if self.annotations and self.current_page_idx < len(self.annotations) - 1:
            self.current_page_idx += 1
            self.current_panel_idx = 0
            self.update_display()
    
    def prev_panel(self):
        if self.annotations:
            current_page = self.annotations[self.current_page_idx]
            if self.current_panel_idx > 0:
                self.current_panel_idx -= 1
                self.update_display()
    
    def next_panel(self):
        if self.annotations:
            current_page = self.annotations[self.current_page_idx]
            total_panels = len(current_page.get("frames", []))
            if self.current_panel_idx < total_panels - 1:
                self.current_panel_idx += 1
                self.update_display()
    
    def reset_to_bbox(self):
        """将当前Panel的classification_points重置为bbox矩形"""
        if not self.annotations:
            return
            
        current_page = self.annotations[self.current_page_idx]
        if self.current_panel_idx >= len(current_page.get("frames", [])):
            return
            
        current_panel = current_page["frames"][self.current_panel_idx]
        if "bbox" in current_panel:
            x1, y1, x2, y2 = current_panel["bbox"]
            # 创建矩形的四个角点（作为classification_points）
            bbox_points = [
                [x1, y1],  # 左上
                [x2, y1],  # 右上
                [x2, y2],  # 右下
                [x1, y2]   # 左下
            ]
            current_panel["classification_points"] = bbox_points  # 更新分类点
            messagebox.showinfo("成功", "已重置为矩形框（更新classification_points）")
            self.update_display()
    
    def start_editing(self):
        """开始编辑当前Panel（加载原图和已标注图）"""
        if not self.annotations:
            messagebox.showwarning("警告", "请先加载标注文件")
            return
            
        current_page = self.annotations[self.current_page_idx]
        if self.current_panel_idx >= len(current_page.get("frames", [])):
            messagebox.showwarning("警告", "没有可编辑的Panel")
            return
        
        # 获取图片文件名（从标注的image_path提取）
        img_basename = os.path.splitext(os.path.basename(current_page["image_path"]))[0]
        # 原图路径（用于编辑）
        original_img_path = os.path.join(ORIGINAL_IMAGE_FOLDER, f"{img_basename}.png")  # 假设原图是png格式
        # 已标注图片路径（用于参考，可选）
        annotated_img_path = os.path.join(ANNOTATED_IMAGE_FOLDER, f"{img_basename}_mask_shapevis.png")
        
        try:
            # 加载原图（必须存在）
            self.image = cv2.imread(original_img_path)
            if self.image is None:
                raise Exception(f"原图不存在: {original_img_path}")
            
            # 尝试加载已标注图片（可选，不存在也不报错）
            self.annotated_image = cv2.imread(annotated_img_path) if os.path.exists(annotated_img_path) else None
                
            # 开始OpenCV编辑器
            self.opencv_editor()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {e}")
    
    def opencv_editor(self):
        """OpenCV图像编辑器（在原图上编辑classification_points）"""
        current_page = self.annotations[self.current_page_idx]
        current_panel = current_page["frames"][self.current_panel_idx]
        
        # 获取或创建classification_points（优先用已有分类点）
        if "classification_points" in current_panel and current_panel["classification_points"]:
            points = np.array(current_panel["classification_points"], dtype=np.float32)
        elif "four_points" in current_panel:
            points = np.array(current_panel["four_points"], dtype=np.float32)
        elif "bbox" in current_panel:
            x1, y1, x2, y2 = current_panel["bbox"]
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        else:
            messagebox.showwarning("警告", "Panel没有可编辑的点")
            return
        
        # 缩放图像以适应屏幕
        h, w = self.image.shape[:2]
        max_size = 1000
        if max(h, w) > max_size:
            self.scale_factor = max_size / max(h, w)
            new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
            self.display_image = cv2.resize(self.image, (new_w, new_h))
            points *= self.scale_factor  # 缩放点坐标
        else:
            self.display_image = self.image.copy()
            self.scale_factor = 1.0
        
        # 设置鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback, points)
        
        print("编辑模式已启动! 使用说明:")
        print("- 拖拽红色圆点调整classification_points位置")
        print("- 按 's' 保存当前调整")
        print("- 按 'r' 重置为矩形框") 
        print("- 按 'n' 保存并切换到下一个Panel")
        print("- 按 'q' 退出编辑")
        
        while True:
            img_display = self.display_image.copy()
            
            # 绘制多边形（基于classification_points）
            pts = points.astype(np.int32)
            cv2.polylines(img_display, [pts], True, (0, 255, 0), 2)  # 绿色多边形
            
            # 绘制可拖拽的点（红色）
            for i, point in enumerate(pts):
                color = (0, 0, 255) if i != self.dragging_point else (255, 0, 0)  # 拖拽时变蓝色
                cv2.circle(img_display, tuple(point), 8, color, -1)
                cv2.putText(img_display, f"P{i}", (point[0]+10, point[1]-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示Panel信息（包含分类点状态）
            shape_type = current_panel.get("shape_type", "Unknown")
            cv2.putText(img_display, 
                       f"Panel {self.current_panel_idx + 1}: {shape_type} (编辑分类点)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imshow(self.window_name, img_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_current_points(points)
            elif key == ord('r'):
                points = self.reset_points_to_bbox(current_panel)
            elif key == ord('n'):
                self.save_current_points(points)
                self.next_panel()
                break
        
        cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, points):
        """鼠标回调函数（拖拽调整点位置）"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击在某个点附近（15像素内）
            for i, point in enumerate(points):
                if np.linalg.norm([x - point[0], y - point[1]]) < 15:
                    self.dragging_point = i
                    break
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point >= 0:
                points[self.dragging_point] = [x, y]  # 实时更新点坐标
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = -1  # 结束拖拽
    
    def save_current_points(self, points):
        """保存当前调整的classification_points（还原缩放）"""
        original_points = points / self.scale_factor  # 还原到原图坐标
        
        current_page = self.annotations[self.current_page_idx]
        current_panel = current_page["frames"][self.current_panel_idx]
        
        # 更新标注中的classification_points
        current_panel["classification_points"] = original_points.tolist()
        print(f"已保存Panel {self.current_panel_idx + 1}的分类点调整")
    
    def reset_points_to_bbox(self, current_panel):
        """重置分类点为bbox矩形（基于原图坐标）"""
        if "bbox" in current_panel:
            x1, y1, x2, y2 = current_panel["bbox"]
            bbox_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            bbox_points *= self.scale_factor  # 缩放适应显示
            print("已重置分类点为矩形框（基于bbox）")
            return bbox_points
        return None
    
    def run(self):
        """运行应用"""
        self.root.mainloop()

if __name__ == "__main__":
    app = InteractiveAnnotationTool()
    app.run()