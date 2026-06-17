"""
将 9dtact 标定文件中的彩色图像转换为黑白（保持 RGB 格式）
转换后 R=G=B，但仍然是三通道
"""

import numpy as np
from pathlib import Path
import shutil

def convert_to_bw_rgb():
    """转换为黑白但保持 RGB 三通道格式"""
    calib_dir = Path("source/tacex_assets/tacex_assets/data/Sensors/9dtact/calibs/640x480")
    npz_file = calib_dir / "dataPack.npz"
    backup_file = calib_dir / "dataPack_backup.npz"
    
    # 如果有备份，先恢复
    if backup_file.exists():
        print(f"发现备份文件，恢复原始数据...")
        shutil.copy2(backup_file, npz_file)
        print(f"✓ 已从备份恢复")
    
    if not npz_file.exists():
        print(f"错误: 文件不存在 {npz_file}")
        return
    
    # 重新备份
    backup_file2 = calib_dir / "dataPack_color_backup.npz"
    if not backup_file2.exists():
        shutil.copy2(npz_file, backup_file2)
        print(f"✓ 已备份彩色版本到: {backup_file2.name}")
    
    # 加载数据
    print(f"\n加载: {npz_file.name}")
    data = np.load(npz_file, allow_pickle=True)
    
    new_data = {}
    
    # 转换 f0 (背景图像) - 保持 RGB 格式
    f0 = data['f0']
    print(f"  f0: {f0.shape} (彩色) -> ", end="")
    # 计算灰度值
    f0_gray_value = (0.299 * f0[:,:,0] + 0.587 * f0[:,:,1] + 0.114 * f0[:,:,2]).astype(np.uint8)
    # 复制到三个通道
    f0_bw = np.stack([f0_gray_value, f0_gray_value, f0_gray_value], axis=2)
    new_data['f0'] = f0_bw
    print(f"{f0_bw.shape} (黑白RGB) ✓")
    
    # 转换 imgs (标定图像批次) - 保持 RGB 格式
    imgs = data['imgs']
    print(f"  imgs: {imgs.shape} (彩色) -> ", end="")
    imgs_bw = np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        gray_value = (0.299 * imgs[i,:,:,0] + 0.587 * imgs[i,:,:,1] + 0.114 * imgs[i,:,:,2]).astype(np.uint8)
        imgs_bw[i,:,:,0] = gray_value
        imgs_bw[i,:,:,1] = gray_value
        imgs_bw[i,:,:,2] = gray_value
    new_data['imgs'] = imgs_bw
    print(f"{imgs_bw.shape} (黑白RGB) ✓")
    
    # 保留其他数据
    for key in data.files:
        if key not in ['f0', 'imgs']:
            new_data[key] = data[key]
            print(f"  {key}: 保持不变")
    
    data.close()
    
    # 保存
    np.savez(npz_file, **new_data)
    print(f"\n✓ 已保存黑白RGB版本到: {npz_file.name}")
    print(f"  彩色版本备份: {backup_file2.name}")
    
    # 验证
    print("\n验证转换结果:")
    verify_data = np.load(npz_file)
    f0_check = verify_data['f0']
    imgs_check = verify_data['imgs']
    print(f"  f0 形状: {f0_check.shape}")
    print(f"  f0 是否黑白 (R=G=B): {np.allclose(f0_check[:,:,0], f0_check[:,:,1]) and np.allclose(f0_check[:,:,1], f0_check[:,:,2])}")
    print(f"  imgs 形状: {imgs_check.shape}")
    print(f"  imgs[0] 是否黑白 (R=G=B): {np.allclose(imgs_check[0,:,:,0], imgs_check[0,:,:,1]) and np.allclose(imgs_check[0,:,:,1], imgs_check[0,:,:,2])}")
    verify_data.close()

if __name__ == "__main__":
    print("="*60)
    print("9dtact 标定图像黑白转换工具")
    print("将彩色图像转换为黑白，但保持 RGB 三通道格式 (R=G=B)")
    print("="*60)
    
    response = input("\n确认转换? (y/n): ")
    
    if response.lower() == 'y':
        convert_to_bw_rgb()
        print("\n完成!")
    else:
        print("取消操作")
