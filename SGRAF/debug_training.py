#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试训练脚本 - 检查损失函数、梯度更新和模型输出
"""

import torch
import pickle
import numpy as np
from model import SGRAF
import data_chinese as data


class SimpleVocab:
    def __init__(self, vocab_dict):
        self.stoi = vocab_dict
        self.itos = {v: k for k, v in vocab_dict.items()}
        
    def __call__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>', 0))
    
    def __len__(self):
        return len(self.stoi)


def debug_training():
    print("=== 调试训练过程 ===\n")
    
    # 配置参数
    opt = type('Args', (), {})()
    opt.data_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/data'
    opt.data_name = 'tk_precomp'
    opt.vocab_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/vocab/tk_precomp_vocab.pkl'
    opt.batch_size = 4  # 小批次便于调试
    opt.workers = 0
    
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
    opt.learning_rate = 0.0002
    
    # 加载词汇表
    with open(opt.vocab_path, 'rb') as f:
        vocab_dict = pickle.load(f)
    vocab = SimpleVocab(vocab_dict)
    opt.vocab_size = len(vocab_dict)
    
    print(f"词汇表大小: {opt.vocab_size}")
    
    # 加载数据
    train_loader = data.get_precomp_loader(
        opt.data_path + '/' + opt.data_name, 'train', vocab, opt,
        batch_size=opt.batch_size, shuffle=True, num_workers=0)
    
    print(f"训练数据批次数: {len(train_loader)}")
    
    # 构建模型
    model = SGRAF(opt)
    
    # 添加logger
    class SimpleLogger:
        def __init__(self):
            self.meters = {}
        def update(self, key, value, n=1):
            if key not in self.meters:
                self.meters[key] = {'val': value, 'sum': value, 'count': n}
            else:
                self.meters[key]['val'] = value
                self.meters[key]['sum'] += value
                self.meters[key]['count'] += n
    
    model.logger = SimpleLogger()
    model.train_start()
    
    print("开始调试训练...")
    
    # 训练几个批次
    for i, (images, captions, lengths, ids) in enumerate(train_loader):
        if i >= 5:  # 只调试前5个批次
            break
            
        print(f"\n=== 批次 {i+1} ===")
        print(f"图像形状: {images.shape}")
        print(f"文本形状: {captions.shape}")
        print(f"文本长度: {lengths[:3]}...")  # 显示前3个长度
        
        # 前向传播
        img_embs, cap_embs, cap_lens = model.forward_emb(images, captions, lengths)
        print(f"图像嵌入形状: {img_embs.shape}")
        print(f"文本嵌入形状: {cap_embs.shape}")
        
        # 计算相似度
        sims = model.forward_sim(img_embs, cap_embs, cap_lens)
        print(f"相似度矩阵形状: {sims.shape}")
        print(f"相似度范围: [{sims.min().item():.4f}, {sims.max().item():.4f}]")
        print(f"对角线相似度: {sims.diag().detach().cpu().numpy()}")
        
        # 计算损失
        loss_before = model.forward_loss(sims)
        print(f"损失值: {loss_before.item():.4f}")
        
        # 检查梯度
        model.optimizer.zero_grad()
        loss_before.backward()
        
        # 检查梯度范数
        total_grad_norm = 0
        param_count = 0
        for name, param in model.txt_enc.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                param_count += 1
                if param_count <= 3:  # 显示前3个参数的梯度
                    print(f"参数 {name}: 梯度范数 = {grad_norm:.6f}")
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"总梯度范数: {total_grad_norm:.6f}")
        
        # 执行优化步骤
        if model.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.params, model.grad_clip)
        model.optimizer.step()
        
        # 再次前向传播检查变化
        with torch.no_grad():
            img_embs2, cap_embs2, cap_lens2 = model.forward_emb(images, captions, lengths)
            sims2 = model.forward_sim(img_embs2, cap_embs2, cap_lens2)
            loss_after = model.criterion(sims2)
            
        print(f"优化后损失: {loss_after.item():.4f}")
        print(f"损失变化: {loss_after.item() - loss_before.item():.6f}")
        
        # 检查参数是否真的更新了
        param_changed = False
        for name, param in model.txt_enc.named_parameters():
            if torch.isnan(param).any():
                print(f"警告: 参数 {name} 包含NaN!")
            if torch.isinf(param).any():
                print(f"警告: 参数 {name} 包含Inf!")
    
    print("\n=== 调试完成 ===")
    print("检查要点:")
    print("1. 损失是否在减少？")
    print("2. 梯度范数是否合理？")
    print("3. 相似度分布是否合理？")
    print("4. 对角线元素是否应该最大？")


if __name__ == '__main__':
    debug_training() 