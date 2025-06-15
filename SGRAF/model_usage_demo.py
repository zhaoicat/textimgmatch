#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中文图文匹配模型使用示例
演示如何加载和使用训练好的模型
"""

import torch
import numpy as np
import pickle
import jieba
import os
import sys

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

def load_model(model_path):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return None, None
    
    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 获取配置和模型
        if 'opt' in checkpoint:
            opt = checkpoint['opt']
            model_state = checkpoint['model']
        else:
            # 如果是直接保存的模型
            print("检测到直接保存的模型文件")
            return checkpoint, None
            
        print(f"✓ 模型加载成功")
        print(f"✓ 训练轮次: {checkpoint.get('epoch', '未知')}")
        print(f"✓ 最佳性能: R@1 = {checkpoint.get('best_r1', '未知')}%")
        
        return model_state, opt
        
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None

def demo_usage():
    """演示模型使用"""
    print("=== 中文图文匹配模型使用演示 ===\n")
    
    # 1. 检查可用的模型文件
    model_paths = [
        "./runs/tk_targeted/checkpoint/best_model.pth",
        "./runs/tk_SGR/checkpoint/model_best.pth.tar",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 未找到训练好的模型文件")
        print("请先运行训练脚本生成模型")
        return
    
    print(f"📁 使用模型文件: {model_path}")
    print(f"📁 文件大小: {os.path.getsize(model_path) / (1024*1024):.1f} MB\n")
    
    # 2. 加载模型
    model_state, opt = load_model(model_path)
    if model_state is None:
        return
    
    print("✅ 演示完成！")
    print(f"📁 最佳模型位置: {model_path}")

if __name__ == "__main__":
    demo_usage() 