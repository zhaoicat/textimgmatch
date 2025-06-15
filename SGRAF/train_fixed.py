#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版本的训练脚本
主要改进：
1. 使用正确的验证机制
2. 改善相似度计算（移除sigmoid限制）
3. 更好的损失监控和早停机制
"""

import os
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.nn.utils import clip_grad_norm_

# 导入数据和模型
import data_chinese as data
from model import SGRAF, ContrastiveLoss
import tensorboard_logger as tb_logger


class SimpleVocab:
    def __init__(self, vocab_dict):
        self.stoi = vocab_dict
        self.itos = {v: k for k, v in vocab_dict.items()}
        
    def __call__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>', 0))
    
    def __len__(self):
        return len(self.stoi)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        return f'{self.val:.4f} ({self.avg:.4f})'


class LogCollector(object):
    def __init__(self):
        self.meters = {}

    def update(self, key, value, n=1):
        if key not in self.meters:
            self.meters[key] = AverageMeter()
        self.meters[key].update(value, n)

    def __str__(self):
        return '  '.join([f'{k}: {str(v)}' for k, v in self.meters.items()])


def compute_retrieval_metrics(sims):
    """计算检索指标"""
    npts = sims.shape[0]
    
    # 图像到文本检索
    i2t_ranks = []
    for i in range(npts):
        inds = np.argsort(sims[i])[::-1]
        rank = np.where(inds == i)[0][0]
        i2t_ranks.append(rank)
    
    # 文本到图像检索  
    t2i_ranks = []
    for i in range(npts):
        inds = np.argsort(sims[:, i])[::-1]
        rank = np.where(inds == i)[0][0]
        t2i_ranks.append(rank)
    
    # 计算召回率
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


def validate_model(opt, val_loader, model):
    """真实的验证函数"""
    model.val_start()
    
    # 提取所有特征
    all_img_embs = []
    all_cap_embs = []
    
    with torch.no_grad():
        for i, (images, captions, lengths, ids) in enumerate(val_loader):
            img_embs, cap_embs, cap_lens = model.forward_emb(images, captions, lengths)
            
            # 平均池化得到全局特征
            img_global = torch.mean(img_embs, dim=1)
            cap_global = torch.mean(cap_embs, dim=1)
            
            all_img_embs.append(img_global.cpu().numpy())
            all_cap_embs.append(cap_global.cpu().numpy())
    
    # 合并所有特征
    img_embs = np.concatenate(all_img_embs, axis=0)
    cap_embs = np.concatenate(all_cap_embs, axis=0)
    
    # L2正则化
    img_embs = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-8)
    cap_embs = cap_embs / (np.linalg.norm(cap_embs, axis=1, keepdims=True) + 1e-8)
    
    # 计算相似度矩阵
    sims = np.dot(img_embs, cap_embs.T)
    
    # 计算检索指标
    metrics = compute_retrieval_metrics(sims)
    
    print(f"验证结果 - I2T: R@1={metrics['i2t_r1']:.1f}, R@5={metrics['i2t_r5']:.1f}, R@10={metrics['i2t_r10']:.1f}")
    print(f"验证结果 - T2I: R@1={metrics['t2i_r1']:.1f}, R@5={metrics['t2i_r5']:.1f}, R@10={metrics['t2i_r10']:.1f}")
    print(f"R-sum: {metrics['rsum']:.1f}")
    
    return metrics['rsum']


def main():
    """主训练函数"""
    print("=== 修复版本训练开始 ===\n")
    
    # 配置参数
    opt = type('Args', (), {})()
    opt.data_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/data'
    opt.data_name = 'tk_precomp'
    opt.vocab_path = '/Users/gszhao/code/小红书/图文匹配/SGRAF/vocab/tk_precomp_vocab.pkl'
    opt.model_name = '/Users/gszhao/code/小红书/图文匹配/SGRAF/runs/tk_fixed/checkpoint'
    opt.logger_name = '/Users/gszhao/code/小红书/图文匹配/SGRAF/runs/tk_fixed/log'
    
    # 训练参数
    opt.batch_size = 16  # 增加批次大小
    opt.num_epochs = 30
    opt.learning_rate = 0.0005  # 提高学习率
    opt.workers = 2
    opt.log_step = 20
    opt.val_step = 100
    opt.grad_clip = 2.0
    opt.margin = 0.2
    opt.max_violation = True
    
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
    opt.lr_update = 15
    opt.temperature = 0.5  # 温度参数
    
    print("配置参数:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")

    # 创建保存目录
    os.makedirs(os.path.dirname(opt.model_name), exist_ok=True)
    os.makedirs(os.path.dirname(opt.logger_name), exist_ok=True)

    # 设置日志
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # 加载词汇表
    print(f"\n加载词汇表：{opt.vocab_path}")
    with open(opt.vocab_path, 'rb') as f:
        vocab_dict = pickle.load(f)
    
    vocab = SimpleVocab(vocab_dict)
    opt.vocab_size = len(vocab_dict)
    print(f"词汇表大小: {opt.vocab_size}")

    # 加载数据
    print("加载数据...")
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)
    
    print(f"训练集批次: {len(train_loader)}")
    print(f"验证集批次: {len(val_loader)}")

    # 构建模型 (使用原始SGRAF但修改相似度计算)
    print("构建模型...")
    model = SGRAF(opt)
    
    # 修改相似度编码器，移除sigmoid
    old_forward = model.sim_enc.forward
    def new_forward(self, img_emb, cap_emb, cap_lens):
        # 调用原始forward但修改最后的sigmoid
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

            # local-global alignment construction
            from model import SCAN_attention, l2norm
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # concat the global and local alignments
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

            # compute the final similarity vector
            if self.module_name == 'SGR':
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)

            # 移除sigmoid，直接输出线性结果
            sim_i = self.sim_eval_w(sim_vec).squeeze(-1)
            # 应用温度缩放
            sim_i = sim_i / opt.temperature
            sim_all.append(sim_i.unsqueeze(1))

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)
        return sim_all
    
    # 替换forward方法
    import types
    model.sim_enc.forward = types.MethodType(new_forward, model.sim_enc)
    
    # 添加logger属性
    model.logger = LogCollector()
    print("模型构建完成")

    # 开始训练
    train_fixed(opt, train_loader, val_loader, model)


def train_fixed(opt, train_loader, val_loader, model):
    """修复版本的训练函数"""
    print("\n开始修复版本训练...")
    best_rsum = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(opt.num_epochs):
        print(f'\n=== Epoch [{epoch+1}/{opt.num_epochs}] ===')
        
        # 调整学习率
        adjust_learning_rate(opt, model.optimizer, epoch)
        
        # 训练一个epoch
        train_epoch(opt, train_loader, model, epoch)
        
        # 验证
        if (epoch + 1) % 2 == 0:  # 每2个epoch验证一次
            print("开始验证...")
            rsum = validate_model(opt, val_loader, model)
            
            # 记录到tensorboard
            tb_logger.log_value('rsum', rsum, step=model.Eiters)
            
            # 保存检查点
            is_best = rsum > best_rsum
            if is_best:
                best_rsum = rsum
                patience_counter = 0
                print(f"🎉 新的最佳模型! R-sum: {best_rsum:.1f}")
            else:
                patience_counter += 1
                print(f"R-sum: {rsum:.1f} (最佳: {best_rsum:.1f})")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'Eiters': model.Eiters,
            }
            
            torch.save(checkpoint, f"{opt.model_name}/checkpoint_{epoch}.pth.tar")
            if is_best:
                torch.save(checkpoint, f"{opt.model_name}/model_best.pth.tar")
            
            # 早停
            if patience_counter >= patience:
                print(f"验证性能连续{patience}次没有提升，早停！")
                break
    
    print(f"\n训练完成！最佳R-sum: {best_rsum:.1f}")


def train_epoch(opt, train_loader, model, epoch):
    """训练一个epoch"""
    model.train_start()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    for i, train_data in enumerate(train_loader):
        # 前向传播
        model.train_emb(*train_data)
        
        # 记录损失
        if hasattr(model.logger, 'meters') and 'Loss' in model.logger.meters:
            losses.update(model.logger.meters['Loss'].val)
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 打印进度
        if i % opt.log_step == 0:
            print(f'Epoch: [{epoch+1}][{i}/{len(train_loader)}] '
                  f'Time: {batch_time.avg:.3f}s '
                  f'Loss: {losses.avg:.4f}')


def adjust_learning_rate(opt, optimizer, epoch):
    """调整学习率"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"学习率调整为: {lr}")


if __name__ == '__main__':
    main() 