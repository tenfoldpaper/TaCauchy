"""
检查和修改 9dtact 标定文件
查看 npz 文件内容并将图像转换为黑白
"""

import numpy as np
from pathlib import Path
import json

def inspect_npz_file(npz_path):
    """检查 npz 文件的内容"""
    print(f"\n{'='*60}")
    print(f"文件: {npz_path.name}")
    print(f"{'='*60}")
    
    data = np.load(npz_path, allow_pickle=True)
    
    for key in data.files:
        arr = data[key]
        print(f"\n键名: {key}")
        print(f"  类型: {type(arr)}")
        print(f"  dtype: {arr.dtype}")
        print(f"  形状: {arr.shape if hasattr(arr, 'shape') else 'N/A'}")
        
        # 如果是图像数据（3D 或 4D 数组）
        if hasattr(arr, 'shape') and len(arr.shape) >= 2:
            if len(arr.shape) == 3 and arr.shape[2] in [3, 4]:
                print(f"  -> 可能是彩色图像 (H={arr.shape[0]}, W={arr.shape[1]}, C={arr.shape[2]})")
            elif len(arr.shape) == 2:
                print(f"  -> 可能是灰度图像 (H={arr.shape[0]}, W={arr.shape[1]})")
            elif len(arr.shape) == 4:
                print(f"  -> 可能是图像批次 (N={arr.shape[0]}, H={arr.shape[1]}, W={arr.shape[2]}, C={arr.shape[3]})")
        
        # 显示数值范围
        if hasattr(arr, 'min') and hasattr(arr, 'max') and arr.dtype != object:
            try:
                if arr.size > 0:
                    print(f"  值范围: [{arr.min():.4f}, {arr.max():.4f}]")
            except:
                pass
    
    data.close()
    return data.files

def convert_to_grayscale(npz_path, output_path=None, backup=True):
    """将 npz 文件中的彩色图像转换为灰度"""
    if output_path is None:
        output_path = npz_path
    
    # 备份原文件
    if backup and output_path == npz_path:
        backup_path = npz_path.parent / f"{npz_path.stem}_backup.npz"
        if not backup_path.exists():
            import shutil
            shutil.copy2(npz_path, backup_path)
            print(f"✓ 已备份到: {backup_path.name}")
    
    data = np.load(npz_path, allow_pickle=True)
    new_data = {}
    converted_count = 0
    
    for key in data.files:
        arr = data[key]
        
        # 检查是否是彩色图像
        if hasattr(arr, 'shape') and len(arr.shape) >= 3:
            # 3D 彩色图像 (H, W, C)
            if len(arr.shape) == 3 and arr.shape[2] in [3, 4]:
                # RGB/RGBA 转灰度
                if arr.shape[2] == 3:
                    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
                else:  # RGBA
                    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
                
                new_data[key] = gray.astype(arr.dtype)
                converted_count += 1
                print(f"✓ 转换 '{key}': {arr.shape} -> {gray.shape}")
            
            # 4D 图像批次 (N, H, W, C)
            elif len(arr.shape) == 4 and arr.shape[3] in [3, 4]:
                gray_batch = []
                for i in range(arr.shape[0]):
                    img = arr[i]
                    if img.shape[2] == 3:
                        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
                    else:  # RGBA
                        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
                    gray_batch.append(gray)
                
                new_data[key] = np.array(gray_batch).astype(arr.dtype)
                converted_count += 1
                print(f"✓ 转换 '{key}': {arr.shape} -> {new_data[key].shape}")
            else:
                new_data[key] = arr
        else:
            new_data[key] = arr
    
    data.close()
    
    # 保存
    if converted_count > 0:
        np.savez(output_path, **new_data)
        print(f"\n✓ 已保存到: {output_path}")
        print(f"  共转换 {converted_count} 个图像数组")
    else:
        print(f"\n! 未发现需要转换的彩色图像")
    
    return converted_count

def main():
    calib_dir = Path("source/tacex_assets/tacex_assets/data/Sensors/9dtact/calibs/640x480")
    
    if not calib_dir.exists():
        print(f"错误: 目录不存在 {calib_dir}")
        return
    
    # 查找所有 npz 文件
    npz_files = list(calib_dir.glob("*.npz"))
    
    print(f"找到 {len(npz_files)} 个 npz 文件")
    
    # 检查每个文件
    for npz_file in npz_files:
        inspect_npz_file(npz_file)
    
    # 询问是否转换
    print(f"\n{'='*60}")
    response = input("\n是否将所有彩色图像转换为灰度? (y/n): ")
    
    if response.lower() == 'y':
        print("\n开始转换...")
        total_converted = 0
        for npz_file in npz_files:
            print(f"\n处理: {npz_file.name}")
            count = convert_to_grayscale(npz_file, backup=True)
            total_converted += count
        
        print(f"\n{'='*60}")
        print(f"完成! 共转换 {total_converted} 个图像数组")
    else:
        print("取消转换")

if __name__ == "__main__":
    main()
