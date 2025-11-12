import scipy.io as sio
import numpy as np
import glob
import os

def safe_squeeze(arr):
    """统一去除冗余维度"""
    arr = np.array(arr)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.squeeze(0)
    return arr

def merge_multiple_mat(mat_files, output_file='merged_all.mat'):
    merged = {}

    for i, mat_file in enumerate(mat_files):
        print(f"🔄 读取文件: {mat_file}")
        data = sio.loadmat(mat_file)

        # 清除系统字段
        for key in ['__header__', '__version__', '__globals__']:
            data.pop(key, None)

        for key, value in data.items():
            value = safe_squeeze(value)

            if key not in merged:
                merged[key] = value
            else:
                try:
                    if merged[key].shape[1:] == value.shape[1:] or \
                       merged[key].ndim == 1 == value.ndim:
                        merged[key] = np.concatenate([merged[key], value], axis=0)
                    else:
                        print(f"[跳过] 字段 {key} 维度不匹配: {merged[key].shape} vs {value.shape}")
                except Exception as e:
                    print(f"[跳过] 字段 {key} 拼接失败: {e}")

    sio.savemat(output_file, merged)
    print(f"\n✅ 合并完成，保存为: {output_file}")

# 示例：合并当前目录下所有 mat 文件
if __name__ == '__main__':
    # mat_files = sorted(glob.glob('./*.mat'))
    mat_files = [
        '/home/sigma/gaze/datasets/gc_mp/metadata_880.mat',
        '/home/sigma/gaze/datasets/gc_mp/metadata_980.mat',
        '/home/sigma/gaze/datasets/gc_mp/metadata_1100.mat',
        '/home/sigma/gaze/datasets/gc_mp/metadata_1220.mat',
        '/home/sigma/gaze/datasets/gc_mp/metadata_1320.mat',
        '/home/sigma/gaze/datasets/gc_mp/metadata_1420.mat',
        '/home/sigma/gaze/datasets/gc_mp/metadata_1472.mat']
    merge_multiple_mat(mat_files, output_file='/home/sigma/gaze/datasets/gc_mp/merged_all.mat')
