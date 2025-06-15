#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高性能训练 - 目标R@1 >= 70%
优化策略: Focal Loss + 硬负例挖掘 + 动态学习率
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.datasets import RawImageDataset
from lib.vse import VSEModel
from data_chinese import get_tokenizer
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector


class OptimizedConfig:
    """优化配置"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk'
        self.data_name = 'tk'
        
        # 模型配置
        self.embed_size = 1024  # 提升嵌入维度
        self.finetune = True
        self.cnn_type = 'resnet152'
        self.use_restval = False
        
        # 训练配置 - 关键优化
        self.batch_size = 24  # 平衡内存和性能
        self.num_epochs = 60
        self.lr_vse = 0.001  # 提高学习率
        self.lr_cnn = 0.0001
        self.lr_decay = 0.8
        self.lr_update = 8
        self.workers = 4
        
        # 损失函数配置
        self.margin = 0.2  # 降低边距
        self.temperature = 0.03  # 非常低的温度
        self.focal_gamma = 2.5  # 增强困难样本关注
        
        # 路径配置
        self.val_step = 1000
        self.log_step = 50
        self.logger_name = './runs/tk_target_70/log'
        self.model_name = './runs/tk_target_70/checkpoint'
        self.resume = ""
        
        # 其他配置
        self.max_violation = True
        self.img_dim = 2048
        self.no_imgnorm = False
        self.reset_train = True
        self.reset_start_epoch = False
        
        # 早停和目标
        self.early_stop_patience = 10
        self.target_r1 = 70.0


class FocalTripletLoss(nn.Module):
    """Focal增强的三元组损失"""
    def __init__(self, margin=0.2, gamma=2.5):
        super(FocalTripletLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        
    def forward(self, im, s):
        batch_size = im.size(0)
        scores = torch.mm(im, s.t())
        diagonal = scores.diag().view(batch_size, 1)
        
        # Image to text
        d1 = diagonal.expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        
        # Text to image  
        d2 = diagonal.t().expand_as(scores)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        
        # 清除对角线
        mask = torch.eye(batch_size) > 0.5
        if torch.cuda.is_available():
            mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        
        # Focal weighting - 增强困难样本
        # 困难样本的损失权重更高
        focal_weight_s = (cost_s / (self.margin + 1e-8)) ** self.gamma
        focal_weight_im = (cost_im / (self.margin + 1e-8)) ** self.gamma
        
        cost_s = cost_s * focal_weight_s
        cost_im = cost_im * focal_weight_im
        
        # 硬负例挖掘
        cost_s = cost_s.max(1)[0] 
        cost_im = cost_im.max(0)[0]
        
        return cost_s.sum() + cost_im.sum()


class AdvancedContrastiveLoss(nn.Module):
    """高级对比损失"""
    def __init__(self, temperature=0.03):
        super(AdvancedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, im, s):
        batch_size = im.size(0)
        
        # 强制标准化
        im = nn.functional.normalize(im, p=2, dim=1)
        s = nn.functional.normalize(s, p=2, dim=1)
        
        # 非常低温度的相似度
        logits = torch.mm(im, s.t()) / self.temperature
        
        labels = torch.arange(batch_size)
        if torch.cuda.is_available():
            labels = labels.cuda()
        
        # 双向交叉熵
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2


def train_epoch(model, data_loader, optimizer, epoch, opt, 
                focal_triplet_loss, contrastive_loss):
    """训练一个epoch"""
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    
    # 动态学习率调整
    if epoch > 0 and epoch % opt.lr_update == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.lr_decay
            print(f"学习率调整为: {param_group['lr']:.6f}")
    
    end = time.time()
    total_batches = len(data_loader)
    
    for i, train_data in enumerate(data_loader):
        data_time.update(time.time() - end)
        
        # 数据准备
        images, captions, lengths, ids = train_data
        
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
        
        # 前向传播
        img_emb, cap_emb = model.forward_emb(images, captions, lengths)
        
        # 计算损失
        loss_focal_triplet = focal_triplet_loss(img_emb, cap_emb)
        loss_contrastive = contrastive_loss(img_emb, cap_emb)
        
        # 组合损失 - 重点关注triplet loss
        total_loss = loss_focal_triplet + 0.6 * loss_contrastive
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        
        optimizer.step()
        
        # 记录
        train_logger.update('Loss', total_loss.item(), img_emb.size(0))
        train_logger.update('FocalTriplet', loss_focal_triplet.item(), img_emb.size(0))
        train_logger.update('Contrastive', loss_contrastive.item(), img_emb.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % opt.log_step == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}][{i}/{total_batches}] '
                  f'LR: {current_lr:.6f} '
                  f'Time {batch_time.val:.3f} '
                  f'Loss {train_logger.meters["Loss"].val:.4f} '
                  f'FT {train_logger.meters["FocalTriplet"].val:.4f} '
                  f'Cont {train_logger.meters["Contrastive"].val:.4f}')


def validate(model, data_loader):
    """验证模型"""
    model.eval()
    
    img_embs = []
    cap_embs = []
    
    print("验证中...")
    with torch.no_grad():
        for val_data in data_loader:
            images, captions, lengths, ids = val_data
            
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)
            
            img_embs.append(img_emb.cpu().numpy())
            cap_embs.append(cap_emb.cpu().numpy())
    
    img_embs = np.concatenate(img_embs, axis=0)
    cap_embs = np.concatenate(cap_embs, axis=0)
    
    # 计算检索指标
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure='cosine')
    (r1i, r5i, r10i, medri, meanri) = t2i(img_embs, cap_embs, measure='cosine')
    
    rsum = r1 + r5 + r10 + r1i + r5i + r10i
    avg_r1 = (r1 + r1i) / 2
    
    print(f"\n验证结果:")
    print(f"Text to Image: R@1: {r1i:.2f}%, R@5: {r5i:.2f}%, R@10: {r10i:.2f}%")
    print(f"Image to Text: R@1: {r1:.2f}%, R@5: {r5:.2f}%, R@10: {r10:.2f}%")
    print(f"平均 R@1: {avg_r1:.2f}%")
    print(f"R-sum: {rsum:.2f}")
    
    return rsum, avg_r1, r1, r1i


def main():
    """主函数"""
    opt = OptimizedConfig()
    
    print("=" * 60)
    print("高性能训练 - 目标R@1 >= 70%")
    print("=" * 60)
    print(f"批次大小: {opt.batch_size}")
    print(f"学习率: {opt.lr_vse}")
    print(f"嵌入维度: {opt.embed_size}")
    print(f"温度参数: {opt.temperature}")
    print(f"边距参数: {opt.margin}")
    print(f"Focal gamma: {opt.focal_gamma}")
    print(f"目标: R@1 >= {opt.target_r1}%")
    print("=" * 60)
    
    # 创建保存目录
    os.makedirs(opt.logger_name, exist_ok=True)
    os.makedirs(opt.model_name, exist_ok=True)
    
    # 数据加载器
    train_loader = DataLoader(
        RawImageDataset(opt.data_path, 'train', get_tokenizer()),
        batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, 
        collate_fn=RawImageDataset.collate_fn
    )
    
    val_loader = DataLoader(
        RawImageDataset(opt.data_path, 'dev', get_tokenizer()),
        batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.workers, 
        collate_fn=RawImageDataset.collate_fn
    )
    
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")
    
    # 模型
    model = VSEModel(opt)
    
    if torch.cuda.is_available():
        model.cuda()
        print("使用GPU训练")
    else:
        print("使用CPU训练")
    
    # 损失函数
    focal_triplet_loss = FocalTripletLoss(
        margin=opt.margin, 
        gamma=opt.focal_gamma
    )
    contrastive_loss = AdvancedContrastiveLoss(
        temperature=opt.temperature
    )
    
    # 优化器
    params = list(model.txt_enc.parameters())
    params += list(model.img_enc.parameters())
    
    optimizer = optim.Adam(params, lr=opt.lr_vse, weight_decay=1e-5)
    
    # 训练状态
    best_rsum = 0
    best_avg_r1 = 0
    patience_counter = 0
    
    print(f"\n开始训练 {opt.num_epochs} epochs...")
    
    for epoch in range(opt.num_epochs):
        print(f"\nEpoch {epoch+1}/{opt.num_epochs}")
        print("-" * 50)
        
        # 训练
        train_epoch(model, train_loader, optimizer, epoch, opt, 
                   focal_triplet_loss, contrastive_loss)
        
        # 验证
        rsum, avg_r1, r1_t2i, r1_i2t = validate(model, val_loader)
        
        # 保存最佳模型
        if avg_r1 > best_avg_r1:
            best_avg_r1 = avg_r1
            best_rsum = rsum
            patience_counter = 0
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'best_avg_r1': best_avg_r1,
                'opt': opt,
            }
            
            model_path = os.path.join(opt.model_name, 'best_model.pth')
            torch.save(checkpoint, model_path)
            
            print(f"💾 保存最佳模型 - 平均R@1: {best_avg_r1:.2f}%")
            
            # 检查是否达到目标
            if best_avg_r1 >= opt.target_r1:
                print(f"🎉 成功达到目标! R@1: {best_avg_r1:.2f}% >= {opt.target_r1}%")
                break
                
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= opt.early_stop_patience:
            print(f"早停触发 - {opt.early_stop_patience}个epoch无改善")
            break
            
        print(f"当前最佳R@1: {best_avg_r1:.2f}% (目标: {opt.target_r1}%)")
    
    # 最终总结
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"最佳平均R@1: {best_avg_r1:.2f}%")
    print(f"最佳R-sum: {best_rsum:.2f}")
    
    if best_avg_r1 >= opt.target_r1:
        print("✅ 成功达到70%目标!")
    else:
        print(f"❌ 未达到目标，当前: {best_avg_r1:.2f}%, 差距: {opt.target_r1 - best_avg_r1:.2f}%")
    
    print(f"{'='*60}")


if __name__ == '__main__':
    main() 