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

def load_vocabulary(vocab_path):
    """加载词汇表"""
    print(f"正在加载词汇表: {vocab_path}")
    
    try:
        if vocab_path.endswith('.pkl'):
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
        elif vocab_path.endswith('.json'):
            import json
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            vocab = vocab_data
        else:
            print("不支持的词汇表格式")
            return None
            
        print(f"✓ 词汇表加载成功，包含 {len(vocab)} 个词汇")
        return vocab
        
    except Exception as e:
        print(f"加载词汇表时出错: {e}")
        return None

def preprocess_text(text, vocab):
    """预处理中文文本"""
    # 使用jieba分词
    tokens = list(jieba.cut(text.strip()))
    
    # 转换为词汇ID
    if hasattr(vocab, 'word2idx'):
        # Vocabulary对象
        word2idx = vocab.word2idx
        caption = [word2idx.get(token, word2idx.get('<unk>', 1)) for token in tokens]
        caption = [word2idx.get('<start>', 0)] + caption + [word2idx.get('<end>', 2)]
    else:
        # 字典格式
        caption = [vocab.get(token, vocab.get('<unk>', 1)) for token in tokens]
        caption = [vocab.get('<start>', 0)] + caption + [vocab.get('<end>', 2)]
    
    return caption, len(caption)

def compute_similarity(image_features, text_features):
    """计算图像和文本特征的相似度"""
    # 归一化特征
    image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    
    # 计算余弦相似度
    similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
    
    return similarity.item()

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
    
    # 3. 加载词汇表
    vocab_paths = [
        "./vocab/tk_precomp_vocab.pkl",
        "./vocab/tk_precomp_vocab.json"
    ]
    
    vocab = None
    for vocab_path in vocab_paths:
        if os.path.exists(vocab_path):
            vocab = load_vocabulary(vocab_path)
            if vocab is not None:
                break
    
    if vocab is None:
        print("❌ 未找到词汇表文件")
        return
    
    print()
    
    # 4. 演示文本预处理
    demo_texts = [
        "这是一幅美丽的唐卡画",
        "佛教艺术作品",
        "传统藏族文化",
        "色彩丰富的宗教画"
    ]
    
    print("🔤 文本预处理演示:")
    for text in demo_texts:
        caption, length = preprocess_text(text, vocab)
        print(f"原文: {text}")
        print(f"分词: {list(jieba.cut(text))}")
        print(f"ID序列: {caption[:10]}... (长度: {length})")
        print()
    
    # 5. 模拟特征计算
    print("🧮 特征相似度计算演示:")
    
    # 模拟图像特征 (1024维)
    image_feature = torch.randn(1024)
    
    # 模拟文本特征 (1024维)  
    text_feature = torch.randn(1024)
    
    # 计算相似度
    similarity = compute_similarity(image_feature, text_feature)
    print(f"图像特征维度: {image_feature.shape}")
    print(f"文本特征维度: {text_feature.shape}")
    print(f"相似度分数: {similarity:.4f}")
    print()
    
    # 6. 使用说明
    print("📖 模型使用说明:")
    print("1. 图像特征: 需要预提取为 (36, 2048) 的区域特征")
    print("2. 文本处理: 使用jieba分词，转换为词汇ID序列")
    print("3. 模型推理: 通过SGRAF模型编码得到特征向量")
    print("4. 相似度计算: 使用余弦相似度匹配图文")
    print()
    
    print("✅ 演示完成！")
    print(f"📁 最佳模型位置: {model_path}")
    print("📚 详细使用方法请参考 '最终验证总结报告.md'")

if __name__ == "__main__":
    demo_usage() 