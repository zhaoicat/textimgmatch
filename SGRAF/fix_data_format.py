#!/usr/bin/env python3
import numpy as np

def expand_to_regions(features, num_regions=36):
    """将2D全局特征扩展为3D区域特征"""
    # 简单重复全局特征，模拟36个区域特征
    expanded = np.tile(features[:, np.newaxis, :], (1, num_regions, 1))
    return expanded

print("修复图像特征格式...")

# 加载当前数据
train_ims = np.load('data/tk_precomp/train_ims.npy') 
dev_ims = np.load('data/tk_precomp/dev_ims.npy')
test_ims = np.load('data/tk_precomp/test_ims.npy')

print('原始形状:', train_ims.shape, dev_ims.shape, test_ims.shape)

# 扩展为区域特征格式 (N, 36, 2048)
train_expanded = expand_to_regions(train_ims)
dev_expanded = expand_to_regions(dev_ims) 
test_expanded = expand_to_regions(test_ims)

print('扩展后形状:', train_expanded.shape, dev_expanded.shape, test_expanded.shape)

# 保存
np.save('data/tk_precomp/train_ims.npy', train_expanded)
np.save('data/tk_precomp/dev_ims.npy', dev_expanded)
np.save('data/tk_precomp/test_ims.npy', test_expanded)

print('数据格式修复完成!') 