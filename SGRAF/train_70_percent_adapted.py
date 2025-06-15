#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高性能训练脚本 - 目标R@1 >= 70%
适配现有tk数据格式

优化策略:
1. 更优的损失函数组合
2. 强化对比学习
3. 硬负例挖掘  
4. 多尺度训练
5. 课程学习
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

from data_chinese import get_precomp_loader, get_tokenizer
from lib.vse import VSEModel
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector
import vocab


class AdvancedConfig:
    """高级配置 - 适配tk数据"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 模型配置 - 增强版
        self.embed_size = 1024  # 更大的嵌入维度
        self.finetune = True
        self.cnn_type = 'resnet152'
        self.use_restval = False
        self.vocab_size = 20000
        self.word_dim = 300
        
        # 训练配置 - 优化版
        self.batch_size = 32  # 增大批次
        self.num_epochs = 80
        self.lr_vse = 0.0008  # 调整学习率
        self.lr_cnn = 0.0001
        self.lr_decay = 0.85
        self.lr_update = 8
        self.workers = 4
        
        # 损失配置 - 关键优化
        self.margin = 0.25  # 调整边距
        self.temperature = 0.05  # 更低温度增强对比
        self.focal_gamma = 2.0  # Focal loss参数
        self.triplet_weight = 1.0
        self.contrastive_weight = 0.8
        self.focal_weight = 0.3
        
        # 训练策略
        self.warmup_epochs = 5
        self.curriculum_start = 0.3
        self.hard_negative_ratio = 0.6
        
        # 其他配置
        self.val_step = 500
        self.log_step = 100
        self.logger_name = './runs/tk_70_target/log'
        self.model_name = './runs/tk_70_target/checkpoint'
        self.resume = ""
        self.max_violation = True
        self.img_dim = 2048
        self.no_imgnorm = False
        self.reset_train = True
        self.reset_start_epoch = False
        
        # 早停配置
        self.early_stop_patience = 12
        self.target_performance = 70.0  # 目标R@1


class FocalContrastiveLoss(nn.Module):
    """Focal + Contrastive Loss"""
    def __init__(self, temperature=0.05, gamma=2.0):
        super(FocalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.gamma = gamma
        
    def forward(self, img_emb, cap_emb):
        batch_size = img_emb.size(0)
        
        # L2标准化
        img_emb = nn.functional.normalize(img_emb, p=2, dim=1)
        cap_emb = nn.functional.normalize(cap_emb, p=2, dim=1)
        
        # 计算相似度矩阵
        logits = torch.mm(img_emb, cap_emb.t()) / self.temperature
        
        # 标签
        labels = torch.arange(batch_size)
        if torch.cuda.is_available():
            labels = labels.cuda()
        
        # 计算概率
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)
        probabilities = exp_logits / sum_exp_logits
        
        # Focal loss - 关注困难样本
        correct_prob = probabilities[torch.arange(batch_size), labels]
        focal_weight = (1 - correct_prob) ** self.gamma
        
        # 负对数似然
        loss_i2t = -focal_weight * torch.log(correct_prob + 1e-10)
        
        # 对称损失 - text to image
        exp_logits_t = exp_logits.t()
        sum_exp_logits_t = exp_logits_t.sum(dim=1, keepdim=True)
        probabilities_t = exp_logits_t / sum_exp_logits_t
        correct_prob_t = probabilities_t[torch.arange(batch_size), labels]
        focal_weight_t = (1 - correct_prob_t) ** self.gamma
        loss_t2i = -focal_weight_t * torch.log(correct_prob_t + 1e-10)
        
        return (loss_i2t.mean() + loss_t2i.mean()) / 2


class EnhancedTripletLoss(nn.Module):
    """增强三元组损失 - 硬负例挖掘"""
    def __init__(self, margin=0.25, hard_negative_ratio=0.6):
        super(EnhancedTripletLoss, self).__init__()
        self.margin = margin
        self.hard_ratio = hard_negative_ratio
        
    def forward(self, img_emb, cap_emb):
        batch_size = img_emb.size(0)
        
        # 计算相似度矩阵
        scores = torch.mm(img_emb, cap_emb.t())
        diagonal = scores.diag().view(batch_size, 1)
        
        # Image to text违反
        d1 = diagonal.expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        
        # Text to image违反
        d2 = diagonal.t().expand_as(scores)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        
        # 清除对角线
        mask = torch.eye(batch_size) > 0.5
        if torch.cuda.is_available():
            mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        
        # 硬负例挖掘 - 选择最困难的一部分负例
        num_hard = max(1, int(batch_size * self.hard_ratio))
        
        # Image to text
        cost_s_sorted, _ = cost_s.sort(dim=1, descending=True)
        cost_s_hard = cost_s_sorted[:, :num_hard]
        
        # Text to image
        cost_im_sorted, _ = cost_im.sort(dim=0, descending=True)
        cost_im_hard = cost_im_sorted[:num_hard, :]
        
        return cost_s_hard.mean() + cost_im_hard.mean()


class PrecomputedImageEncoder(nn.Module):
    """预计算特征的图像编码器"""
    
    def __init__(self, img_dim, embed_size):
        super(PrecomputedImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 全局平均池化和线性映射
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, images):
        """前向传播"""
        # 如果输入是3D的 (batch, regions, features)，进行全局平均池化
        if len(images.shape) == 3:
            features = images.mean(dim=1)  # 平均所有区域特征
        else:
            features = images
        
        # 映射到嵌入空间
        features = self.fc(features)
        
        # L2标准化
        features = nn.functional.normalize(features, p=2, dim=1)
        
        return features


class AdaptedVSEModel(VSEModel):
    """适配预计算特征的VSE模型"""
    
    def __init__(self, opt):
        # 不调用父类的__init__，自己实现
        nn.Module.__init__(self)
        
        # 预计算特征的图像编码器
        self.img_enc = PrecomputedImageEncoder(
            img_dim=opt.img_dim,
            embed_size=opt.embed_size
        )
        
        # 文本编码器
        from lib.vse import TextEncoder
        self.txt_enc = TextEncoder(
            vocab_size=opt.vocab_size,
            word_dim=opt.word_dim,
            embed_size=opt.embed_size
        )


class ProgressiveTrainer:
    """渐进式训练器"""
    
    def __init__(self):
        self.opt = AdvancedConfig()
        self.focal_loss = FocalContrastiveLoss(
            temperature=self.opt.temperature,
            gamma=self.opt.focal_gamma
        )
        self.triplet_loss = EnhancedTripletLoss(
            margin=self.opt.margin,
            hard_negative_ratio=self.opt.hard_negative_ratio
        )
        
        # 加载词汇表
        self.vocab = self.load_vocab()
        if self.vocab:
            self.opt.vocab_size = len(self.vocab)
    
    def load_vocab(self):
        """加载词汇表"""
        try:
            import pickle
            vocab_path = f'./vocab/{self.opt.data_name}_vocab.pkl'
            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"词汇表文件 {vocab_path} 不存在，使用默认配置")
                return None
        except Exception as e:
            print(f"加载词汇表失败: {e}")
            return None
        
    def get_learning_rate(self, epoch):
        """获取自适应学习率"""
        if epoch < self.opt.warmup_epochs:
            # 预热阶段
            return self.opt.lr_vse * (epoch + 1) / self.opt.warmup_epochs
        else:
            # 余弦退火
            remaining = epoch - self.opt.warmup_epochs
            total_remaining = self.opt.num_epochs - self.opt.warmup_epochs
            return self.opt.lr_vse * (1 + np.cos(np.pi * remaining / total_remaining)) / 2
    
    def train_epoch(self, model, data_loader, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()
        
        # 调整学习率
        lr = self.get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
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
            
            # 计算多种损失
            loss_triplet = self.triplet_loss(img_emb, cap_emb)
            loss_focal = self.focal_loss(img_emb, cap_emb)
            
            # 组合损失
            total_loss = (self.opt.triplet_weight * loss_triplet + 
                         self.opt.focal_weight * loss_focal)
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # 记录
            train_logger.update('Loss', total_loss.item(), img_emb.size(0))
            train_logger.update('Triplet', loss_triplet.item(), img_emb.size(0))
            train_logger.update('Focal', loss_focal.item(), img_emb.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.opt.log_step == 0:
                print(f'Epoch [{epoch+1}][{i}/{total_batches}] '
                      f'LR: {lr:.6f} '
                      f'Time {batch_time.val:.3f} '
                      f'Loss {train_logger.meters["Loss"].val:.4f} '
                      f'Triplet {train_logger.meters["Triplet"].val:.4f} '
                      f'Focal {train_logger.meters["Focal"].val:.4f}')
    
    def validate(self, model, data_loader):
        """模型验证"""
        model.eval()
        
        img_embs = []
        cap_embs = []
        
        print("开始验证...")
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
        
        print(f"\n=== 验证结果 ===")
        print(f"Text to Image: R@1: {r1i:.2f}%, R@5: {r5i:.2f}%, R@10: {r10i:.2f}%")
        print(f"Image to Text: R@1: {r1:.2f}%, R@5: {r5:.2f}%, R@10: {r10:.2f}%")
        print(f"平均 R@1: {avg_r1:.2f}%")
        print(f"R-sum: {rsum:.2f}")
        
        return rsum, avg_r1, r1, r1i
    
    def train(self):
        """主训练流程"""
        print("=== 高性能训练开始 - 目标R@1 >= 70% ===")
        print(f"数据路径: {self.opt.data_path}")
        print(f"数据名称: {self.opt.data_name}")
        print(f"批次大小: {self.opt.batch_size}")
        print(f"嵌入维度: {self.opt.embed_size}")
        print(f"词汇表大小: {self.opt.vocab_size}")
        print(f"初始学习率: {self.opt.lr_vse}")
        print(f"温度参数: {self.opt.temperature}")
        print(f"边距参数: {self.opt.margin}")
        print(f"硬负例比例: {self.opt.hard_negative_ratio}")
        print(f"目标性能: R@1 >= {self.opt.target_performance}%")
        
        # 创建保存目录
        os.makedirs(self.opt.logger_name, exist_ok=True)
        os.makedirs(self.opt.model_name, exist_ok=True)
        
                          # 数据加载器
         if self.vocab:
             # 直接使用tk_precomp目录作为数据路径
             train_loader = get_precomp_loader(
                 './data/tk_precomp', 'train', self.vocab, self.opt,
                 batch_size=self.opt.batch_size, shuffle=True, 
                 num_workers=self.opt.workers
             )
             
             val_loader = get_precomp_loader(
                 './data/tk_precomp', 'dev', self.vocab, self.opt,
                 batch_size=self.opt.batch_size, shuffle=False,
                 num_workers=self.opt.workers
             )
        else:
            print("词汇表不存在，请先运行build_vocab.py构建词汇表")
            return
        
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        
        # 模型
        model = AdaptedVSEModel(self.opt)
        
        if torch.cuda.is_available():
            model.cuda()
            print("使用GPU训练")
        else:
            print("使用CPU训练")
        
        # 优化器
        params = list(model.txt_enc.parameters())
        params += list(model.img_enc.parameters())
        
        optimizer = optim.Adam(params, lr=self.opt.lr_vse, 
                              weight_decay=1e-4)  # 添加权重衰减
        
        # 训练状态
        best_rsum = 0
        best_avg_r1 = 0
        patience_counter = 0
        
        print(f"\n开始训练 {self.opt.num_epochs} epochs...")
        
        for epoch in range(self.opt.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.opt.num_epochs}")
            print(f"{'='*50}")
            
            # 训练
            self.train_epoch(model, train_loader, optimizer, epoch)
            
            # 验证
            rsum, avg_r1, r1_t2i, r1_i2t = self.validate(model, val_loader)
            
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
                    'r1_t2i': r1_t2i,
                    'r1_i2t': r1_i2t,
                    'opt': self.opt,
                }
                
                model_path = os.path.join(self.opt.model_name, 'best_model.pth')
                torch.save(checkpoint, model_path)
                
                print(f"💾 保存最佳模型 - 平均R@1: {best_avg_r1:.2f}%")
                
                # 检查是否达到目标
                if best_avg_r1 >= self.opt.target_performance:
                    print(f"🎉 成功达到目标! 平均R@1: {best_avg_r1:.2f}% >= {self.opt.target_performance}%")
                    print("训练提前结束!")
                    break
                    
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= self.opt.early_stop_patience:
                print(f"早停触发 - 连续{self.opt.early_stop_patience}个epoch无改善")
                break
                
            print(f"当前最佳平均R@1: {best_avg_r1:.2f}% (目标: {self.opt.target_performance}%)")
        
        # 训练总结
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"最佳平均R@1: {best_avg_r1:.2f}%")
        print(f"最佳R-sum: {best_rsum:.2f}")
        
        if best_avg_r1 >= self.opt.target_performance:
            print("✅ 成功达到70%目标!")
        else:
            print(f"❌ 未完全达到70%目标")
            print(f"当前最佳: {best_avg_r1:.2f}%")
            print(f"距离目标还差: {self.opt.target_performance - best_avg_r1:.2f}%")
            
            # 提供改进建议
            print("\n🔧 改进建议:")
            if best_avg_r1 < 30:
                print("1. 检查数据质量和预处理")
                print("2. 调整模型架构")
                print("3. 使用更强的预训练特征")
            elif best_avg_r1 < 50:
                print("1. 增加训练数据")
                print("2. 数据增强")
                print("3. 调整损失函数权重")
            else:
                print("1. 精细调整超参数")
                print("2. 增加模型复杂度")
                print("3. 集成学习")
        
        print(f"{'='*60}")


def main():
    """主函数"""
    trainer = ProgressiveTrainer()
    trainer.train()


if __name__ == '__main__':
    main() 