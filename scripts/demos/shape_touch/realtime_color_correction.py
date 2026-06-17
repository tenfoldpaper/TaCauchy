#!/usr/bin/env python3
"""实时颜色校正模块 - 用于9DTact传感器"""

import numpy as np
import cv2
import torch
from pathlib import Path


class RealtimeColorCorrector:
    """实时颜色校正器，使渲染图与背景图色调一致"""
    
    def __init__(self, background_image_path: str, window_name: str = "9DTact Corrected", save_dir: str = None):
        """
        初始化颜色校正器
        
        Args:
            background_image_path: 背景图像路径 (0.png)
            window_name: OpenCV窗口名称
            save_dir: 保存图像的目录路径
        """
        self.window_name = window_name
        self.save_dir = Path(save_dir) if save_dir else Path("IROS2026_Template/simimages")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载背景图像
        bg_img = cv2.imread(background_image_path)
        if bg_img is None:
            raise FileNotFoundError(f"无法加载背景图像: {background_image_path}")
        
        bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # 计算背景图像的统计信息
        self.bg_mean = bg_rgb.mean(axis=(0, 1))  # Shape: (3,)
        self.bg_std = bg_rgb.std(axis=(0, 1))    # Shape: (3,)
        
        print(f"[ColorCorrector] 背景图统计:")
        print(f"  均值: R={self.bg_mean[0]:.2f}, G={self.bg_mean[1]:.2f}, B={self.bg_mean[2]:.2f}")
        print(f"  标准差: R={self.bg_std[0]:.2f}, G={self.bg_std[1]:.2f}, B={self.bg_std[2]:.2f}")
        print(f"[ColorCorrector] 图像保存目录: {self.save_dir}")
        
        # 创建OpenCV窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 设置鼠标回调
        self.current_image = None
        self.save_counter = 0
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self.frame_count = 0
        
        print(f"[ColorCorrector] 提示: 在 '{self.window_name}' 窗口中点击鼠标左键保存当前图像")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数 - 左键点击保存图像"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_image is not None:
                self._save_image(self.current_image)
    
    def _save_image(self, image: np.ndarray):
        """保存图像到文件"""
        # 确保图像是640x480分辨率
        if image.shape[:2] != (480, 640):
            image_resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        else:
            image_resized = image
        
        # 生成文件名
        filename = f"tactile_image_{self.save_counter:04d}.png"
        filepath = self.save_dir / filename
        
        # 保存图像 (OpenCV expects BGR format)
        success = cv2.imwrite(str(filepath), image_resized)
        
        if success:
            self.save_counter += 1
            print(f"[ColorCorrector] ✓ 图像已保存: {filepath} (640x480)")
        else:
            print(f"[ColorCorrector] ✗ 保存失败: {filepath}")
    
    def correct_color_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        应用颜色校正 (NumPy版本)
        
        Args:
            image: 输入图像 (H, W, 3), RGB格式, 值域 [0, 255] 或 [0, 1]
            
        Returns:
            校正后的图像 (H, W, 3), RGB格式, 值域 [0, 255]
        """
        # 确保是float32
        if image.dtype == np.uint8:
            img = image.astype(np.float32)
        else:
            img = image.astype(np.float32)
            # 如果值域是[0, 1]，转换为[0, 255]
            if img.max() <= 1.0:
                img = img * 255.0
        
        # 计算当前图像的统计信息
        img_mean = img.mean(axis=(0, 1))  # Shape: (3,)
        img_std = img.std(axis=(0, 1))    # Shape: (3,)
        
        # 应用均值-标准差匹配
        corrected = np.zeros_like(img)
        for c in range(3):
            # 避免除零
            std_safe = max(img_std[c], 1e-6)
            # 归一化到零均值、单位方差
            normalized = (img[:, :, c] - img_mean[c]) / std_safe
            # 缩放和平移到背景分布
            corrected[:, :, c] = normalized * self.bg_std[c] + self.bg_mean[c]
        
        # 裁剪到有效范围 [0, 255]
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
    
    def correct_color_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        应用颜色校正 (PyTorch版本)
        
        Args:
            image: 输入图像 (H, W, 3), RGB格式, 值域 [0, 1]
            
        Returns:
            校正后的图像 (H, W, 3), RGB格式, 值域 [0, 1]
        """
        # 转换为[0, 255]范围
        img = image * 255.0
        
        # 转换为numpy进行处理
        img_np = img.cpu().numpy()
        
        # 应用校正
        corrected_np = self.correct_color_numpy(img_np)
        
        # 转换回torch并归一化到[0, 1]
        corrected = torch.from_numpy(corrected_np).to(image.device).float() / 255.0
        
        return corrected
    
    def show_corrected_image(self, image: np.ndarray, wait_time: int = 1):
        """
        显示校正后的图像
        
        Args:
            image: 输入图像 (H, W, 3), RGB格式
            wait_time: OpenCV waitKey时间 (ms)
        """
        # 应用颜色校正
        corrected = self.correct_color_numpy(image)
        
        # 转换为BGR用于OpenCV显示
        corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
        
        # 保存当前图像引用（用于鼠标点击保存）
        self.current_image = corrected_bgr.copy()
        
        # 在图像上添加保存提示
        display_image = corrected_bgr.copy()
        text = "Click to Save (640x480)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # 获取文本大小
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 在右上角绘制半透明背景
        padding = 10
        bg_x1 = display_image.shape[1] - text_width - padding * 2
        bg_y1 = padding
        bg_x2 = display_image.shape[1] - padding
        bg_y2 = text_height + padding * 2
        
        # 创建半透明背景
        overlay = display_image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display_image, 0.5, 0, display_image)
        
        # 绘制文本
        text_x = bg_x1 + padding
        text_y = bg_y1 + text_height + padding // 2
        cv2.putText(display_image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        # 显示
        cv2.imshow(self.window_name, display_image)
        cv2.waitKey(wait_time)
        
        self.frame_count += 1
        
        # 每100帧打印一次统计信息
        if self.frame_count % 100 == 0:
            corrected_mean = corrected.mean(axis=(0, 1))
            print(f"[ColorCorrector] Frame {self.frame_count}: "
                  f"R={corrected_mean[0]:.1f}, G={corrected_mean[1]:.1f}, B={corrected_mean[2]:.1f}")
    
    def close(self):
        """关闭OpenCV窗口"""
        cv2.destroyWindow(self.window_name)


# 测试代码
if __name__ == "__main__":
    from pathlib import Path
    
    # 路径
    calib_folder = Path("source/tacex_assets/tacex_assets/data/Sensors/9dtact/calibs/640x480")
    bg_path = calib_folder / "0.png"
    render_path = calib_folder / "render.png"
    
    print("=" * 80)
    print("实时颜色校正测试")
    print("=" * 80)
    
    # 创建校正器
    corrector = RealtimeColorCorrector(str(bg_path))
    
    # 加载渲染图
    render_img = cv2.imread(str(render_path))
    if render_img is None:
        print(f"错误: 无法加载渲染图 {render_path}")
        exit(1)
    
    render_rgb = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
    
    print(f"\n渲染图形状: {render_rgb.shape}")
    render_mean = render_rgb.mean(axis=(0, 1))
    print(f"渲染图均值: R={render_mean[0]:.2f}, G={render_mean[1]:.2f}, B={render_mean[2]:.2f}")
    
    # 应用校正
    corrected = corrector.correct_color_numpy(render_rgb)
    
    print(f"\n校正后形状: {corrected.shape}")
    corrected_mean = corrected.mean(axis=(0, 1))
    print(f"校正后均值: R={corrected_mean[0]:.2f}, G={corrected_mean[1]:.2f}, B={corrected_mean[2]:.2f}")
    
    # 保存对比图
    bg_img = cv2.imread(str(bg_path))
    bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    
    # 调整大小以匹配
    if bg_rgb.shape != render_rgb.shape:
        render_rgb_resized = cv2.resize(render_rgb, (bg_rgb.shape[1], bg_rgb.shape[0]))
        corrected_resized = cv2.resize(corrected, (bg_rgb.shape[1], bg_rgb.shape[0]))
    else:
        render_rgb_resized = render_rgb
        corrected_resized = corrected
    
    comparison = np.hstack([bg_rgb, render_rgb_resized, corrected_resized])
    cv2.imwrite(
        str(calib_folder / "realtime_correction_test.png"),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    )
    print(f"\n✓ 保存对比图: {calib_folder / 'realtime_correction_test.png'}")
    print("  (背景图 | 原始渲染 | 校正后)")
    
    # 显示图像
    print("\n按任意键关闭窗口...")
    cv2.imshow("Background", cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("Original Render", cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("Corrected", cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
