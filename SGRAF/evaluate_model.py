#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 测试训练好的SGRAF模型
"""

import os
import pickle
import torch
import numpy as np
from model import SGRAF
import data_chinese as data


class SimpleVocab:
    """简单的词汇表类"""
    def __init__(self, vocab_dict):
        self.stoi = vocab_dict
        self.itos = {v: k for k, v in vocab_dict.items()}
        
    def __call__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>', 0))
    
    def __len__(self):
        return len(self.stoi)


def compute_similarity_matrix(img_embs, cap_embs):
    """计算图像和文本嵌入的相似度矩阵"""
    # 平均池化得到全局特征
    if len(img_embs.shape) == 3:
        img_global = np.mean(img_embs, axis=1)
    else:
        img_global = img_embs
        
    if len(cap_embs.shape) == 3:
        cap_global = np.mean(cap_embs, axis=1)
    else:
        cap_global = cap_embs
    
    # L2正则化
    img_global = img_global / (np.linalg.norm(img_global, axis=1, keepdims=True) + 1e-8)
    cap_global = cap_global / (np.linalg.norm(cap_global, axis=1, keepdims=True) + 1e-8)
    
    # 计算相似度矩阵
    similarity_matrix = np.dot(img_global, cap_global.T)
    return similarity_matrix


def compute_retrieval_metrics(similarity_matrix):
    """计算检索指标"""
    n_samples = similarity_matrix.shape[0]
    
    # 图像到文本检索
    i2t_ranks = []
    for i in range(n_samples):
        scores = similarity_matrix[i]
        # 按相似度排序
        sorted_indices = np.argsort(scores)[::-1]
        # 找到正确匹配的排名
        rank = np.where(sorted_indices == i)[0][0]
        i2t_ranks.append(rank)
    
    # 文本到图像检索
    t2i_ranks = []
    for i in range(n_samples):
        scores = similarity_matrix[:, i]
        # 按相似度排序
        sorted_indices = np.argsort(scores)[::-1]
        # 找到正确匹配的排名
        rank = np.where(sorted_indices == i)[0][0]
        t2i_ranks.append(rank)
    
    # 计算召回率指标
    i2t_ranks = np.array(i2t_ranks)
    t2i_ranks = np.array(t2i_ranks)
    
    # R@1, R@5, R@10
    i2t_r1 = 100.0 * len(np.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
    i2t_r5 = 100.0 * len(np.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
    i2t_r10 = 100.0 * len(np.where(i2t_ranks < 10)[0]) / len(i2t_ranks)
    
    t2i_r1 = 100.0 * len(np.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
    t2i_r5 = 100.0 * len(np.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
    t2i_r10 = 100.0 * len(np.where(t2i_ranks < 10)[0]) / len(t2i_ranks)
    
    return {
        'i2t_r1': i2t_r1, 'i2t_r5': i2t_r5, 'i2t_r10': i2t_r10,
        't2i_r1': t2i_r1, 't2i_r5': t2i_r5, 't2i_r10': t2i_r10,
        'rsum': i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
    }


def evaluate_model():
    """评估模型"""
    print("=== 唐卡图文匹配模型评估 ===\n")
    
    # 配置参数
    opt = type('Args', (), {})()
    opt.data_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/data'
    opt.data_name = 'tk_precomp'
    opt.vocab_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/vocab/tk_precomp_vocab.pkl'
    opt.batch_size = 100
    opt.workers = 2
    
    # 模型参数
    opt.img_dim = 2048
    opt.word_dim = 300
    opt.embed_size = 1024
    opt.sim_dim = 256
    opt.num_layers = 1
    opt.bi_gru = True
    opt.no_imgnorm = False
    opt.no_txtnorm = False
    opt.module_name = 'SGR'
    opt.sgr_step = 3
    opt.grad_clip = 2.0
    opt.margin = 0.2
    opt.max_violation = True
    
    # 加载词汇表
    print("加载词汇表...")
    with open(opt.vocab_path, 'rb') as f:
        vocab_dict = pickle.load(f)
    vocab = SimpleVocab(vocab_dict)
    opt.vocab_size = len(vocab_dict)
    
    # 构建模型
    print("加载模型...")
    model = SGRAF(opt)
    
    # 加载训练好的模型权重
    model_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/runs/tk_SGR/checkpoint/model_best.pth.tar'
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.val_start()
    
    print(f"模型加载完成，训练轮数: {checkpoint['epoch']}")
    print(f"最佳R-sum: {checkpoint['best_rsum']:.1f}\n")
    
    # 加载测试数据
    print("加载测试数据...")
    test_loader = data.get_precomp_loader(
        opt.data_path + '/' + opt.data_name, 'test', vocab, opt,
        batch_size=100, shuffle=False, num_workers=2)
    
    # 提取所有图像和文本特征
    print("提取图像和文本特征...")
    all_img_embs = []
    all_cap_embs = []
    all_cap_lens = []
    
    with torch.no_grad():
        for i, (images, captions, lengths, ids) in enumerate(test_loader):
            img_embs, cap_embs, cap_lens = model.forward_emb(images, captions, lengths)
            all_img_embs.append(img_embs.cpu().numpy())
            all_cap_embs.append(cap_embs.cpu().numpy())
            all_cap_lens.extend(cap_lens)
            
            print(f"处理batch {i+1}/{len(test_loader)}")
    
    # 合并所有特征
    all_img_embs = np.concatenate(all_img_embs, axis=0)
    all_cap_embs = np.concatenate(all_cap_embs, axis=0)
    
    print(f"图像特征形状: {all_img_embs.shape}")
    print(f"文本特征形状: {all_cap_embs.shape}")
    print(f"测试样本数: {len(all_cap_lens)}\n")
    
    # 计算相似度矩阵
    print("计算相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(all_img_embs, all_cap_embs)
    print(f"相似度矩阵形状: {similarity_matrix.shape}\n")
    
    # 计算检索指标
    print("计算检索指标...")
    metrics = compute_retrieval_metrics(similarity_matrix)
    
    # 输出结果
    print("=== 评估结果 ===")
    print(f"图像到文本检索:")
    print(f"  R@1:  {metrics['i2t_r1']:.2f}%")
    print(f"  R@5:  {metrics['i2t_r5']:.2f}%")
    print(f"  R@10: {metrics['i2t_r10']:.2f}%")
    print()
    print(f"文本到图像检索:")
    print(f"  R@1:  {metrics['t2i_r1']:.2f}%")
    print(f"  R@5:  {metrics['t2i_r5']:.2f}%")
    print(f"  R@10: {metrics['t2i_r10']:.2f}%")
    print()
    print(f"总分 (R-sum): {metrics['rsum']:.2f}")
    
    # 展示一些示例
    print("\n=== 检索示例 ===")
    show_retrieval_examples(similarity_matrix, test_loader, 3)
    
    return metrics


def show_retrieval_examples(similarity_matrix, test_loader, num_examples=3):
    """展示一些检索示例"""
    # 获取文本数据
    captions = []
    with open('/Users/gszhao/code/小红书/图文匹配/SGRAF/data/tk_precomp/test_caps.txt', 'r', encoding='utf-8') as f:
        for line in f:
            captions.append(line.strip())
    
    for i in range(min(num_examples, len(captions))):
        print(f"\n示例 {i+1}:")
        print(f"原文本: {captions[i]}")
        
        # 找到最相似的图像
        scores = similarity_matrix[:, i]
        top_matches = np.argsort(scores)[::-1][:3]
        
        print("最相似的图像排名:")
        for j, match_idx in enumerate(top_matches):
            print(f"  {j+1}. 图像{match_idx} (相似度: {scores[match_idx]:.4f})")
            if match_idx == i:
                print("     ✓ 正确匹配!")


if __name__ == '__main__':
    evaluate_model() 