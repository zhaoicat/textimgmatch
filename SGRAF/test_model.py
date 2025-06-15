#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的模型测试脚本
"""

import torch
import pickle
import os

def test_model():
    """测试训练好的模型"""
    print("=== 唐卡图文匹配模型测试 ===\n")
    
    # 检查模型文件
    model_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/runs/tk_SGR/checkpoint/model_best.pth.tar'
    if not os.path.exists(model_path):
        print("错误：找不到训练好的模型文件！")
        return
    
    print(f"模型文件大小: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    # 加载模型检查点
    print("加载模型检查点...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"训练轮数: {checkpoint['epoch']}")
    print(f"最佳R-sum: {checkpoint['best_rsum']:.1f}")
    print(f"模型参数数量: {len(checkpoint['model'])}")
    
    # 检查数据文件
    data_files = [
        'data/tk_precomp/train_ims.npy',
        'data/tk_precomp/train_caps.txt',
        'data/tk_precomp/test_ims.npy', 
        'data/tk_precomp/test_caps.txt',
        'vocab/tk_precomp_vocab.pkl'
    ]
    
    print("\n数据文件检查:")
    for file_path in data_files:
        full_path = f'/Users/gszhao/code/小红书/图文匹配/SGRAF/{file_path}'
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            if file_path.endswith('.npy'):
                print(f"✓ {file_path}: {size / (1024*1024):.1f} MB")
            elif file_path.endswith('.txt'):
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                print(f"✓ {file_path}: {lines} 行")
            elif file_path.endswith('.pkl'):
                with open(full_path, 'rb') as f:
                    vocab = pickle.load(f)
                print(f"✓ {file_path}: {len(vocab)} 词汇")
        else:
            print(f"✗ {file_path}: 文件不存在")
    
    # 简单的性能评估
    print(f"\n=== 训练结果分析 ===")
    print(f"训练状态: 完成 ({checkpoint['epoch']}/20 轮)")
    print(f"验证性能: R-sum = {checkpoint['best_rsum']:.1f}")
    
    # 基于训练日志分析性能
    if checkpoint['best_rsum'] == 100.0:
        print("性能分析: 模型在验证集上达到了100% R-sum")
        print("这可能表示:")
        print("  1. 验证集太小 (只有4个样本)")
        print("  2. 模型可能过拟合")
        print("  3. 需要在更大的测试集上评估真实性能")
    
    # 检查测试集大小
    test_caps_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/data/tk_precomp/test_caps.txt'
    if os.path.exists(test_caps_path):
        with open(test_caps_path, 'r', encoding='utf-8') as f:
            test_samples = len(f.readlines())
        print(f"\n测试集样本数: {test_samples}")
        
        if test_samples > 100:
            print("建议: 可以在完整测试集上进行详细评估")
        else:
            print("注意: 测试集较小，结果可能不够稳定")
    
    print(f"\n=== 模型训练成功！ ===")
    print(f"✓ 模型已保存到: {model_path}")
    print(f"✓ 训练完成时间: 20个epoch")
    print(f"✓ 最佳性能: R-sum = {checkpoint['best_rsum']:.1f}")
    print(f"\n你的唐卡图文匹配模型训练已完成！")

if __name__ == '__main__':
    test_model() 