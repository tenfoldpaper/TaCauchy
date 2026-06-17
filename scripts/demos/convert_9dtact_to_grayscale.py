"""
将 9dtact 标定文件中的彩色图像转换为灰度
"""

import numpy as np
from pathlib import Path
import shutil

def convert_to_grayscale():
    """转换标定文件中的图像为灰度"""
    calib_dir = Path("source/tacex_assets/tacex_assets/data/Sensors/9dtact/calibs/640x480")
    npz_file = calib_dir / "dataPack.npz"
    
    if not npz_file.exists():
        print(f"错误: 文件不存在 {npz_file}")
        return
    
    # 备份
    backup_file = calib_dir / "dataPack_backup.npz"
    if not backup_file.exists():
        shutil.copy2(npz_file, backup_file)
        print(f"✓ 已备份到: {backup_file.name}")
    
    # 加载数据
    print(f"\n加载: {npz_file.name}")
    data = np.load(npz_file, allow_pickle=True)
    
    new_data = {}
    
    # 转换 f0 (背景图像)
    f0 = data['f0']
    print(f"  f0: {f0.shape} -> ", end="")
    f0_gray = (0.299 * f0[:,:,0] + 0.587 * f0[:,:,1] + 0.114 * f0[:,:,2]).astype(np.uint8)
    new_data['f0'] = f0_gray
    print(f"{f0_gray.shape} ✓")
    
    # 转换 imgs (标定图像批次)
    imgs = data['imgs']
    print(f"  imgs: {imgs.shape} -> ", end="")
    imgs_gray = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_gray[i] = (0.299 * imgs[i,:,:,0] + 0.587 * imgs[i,:,:,1] + 0.114 * imgs[i,:,:,2]).astype(np.uint8)
    new_data['imgs'] = imgs_gray
    print(f"{imgs_gray.shape} ✓")
    
    # 保留其他数据
    for key in data.files:
        if key not in ['f0', 'imgs']:
            new_data[key] = data[key]
            print(f"  {key}: 保持不变")
    
    data.close()
    
    # 保存
    np.savez(npz_file, **new_data)
    print(f"\n✓ 已保存灰度版本到: {npz_file.name}")
    print(f"  原文件备份: {backup_file.name}")
    
    # 验证
    print("\n验证转换结果:")
    verify_data = np.load(npz_file)
    print(f"  f0 形状: {verify_data['f0'].shape}")
    print(f"  imgs 形状: {verify_data['imgs'].shape}")
    verify_data.close()

if __name__ == "__main__":
    print("="*60)
    print("9dtact 标定图像灰度转换工具")
    print("="*60)
    
    response = input("\n确认将彩色图像转换为灰度? (y/n): ")
    
    if response.lower() == 'y':
        convert_to_grayscale()
        print("\n完成!")
    else:
        print("取消操作")
