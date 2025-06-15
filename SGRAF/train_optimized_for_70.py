#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对低R@1性能的全面优化训练脚本

问题分析：之前R-sum只有9.37，说明存在严重问题
优化策略：
1. 修复数据预处理和加载问题
2. 改进模型架构（注意力机制、多头注意力等）
3. 优化损失函数（InfoNCE、难负例挖掘等）
4. 改进训练策略（学习率调度、数据增强等）
5. 使用更强的预训练特征
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import math

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

from data_chinese import get_precomp_loader
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector
import vocab


class OptimizedConfig:
    """优化配置 - 针对低性能问题"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 模型配置 - 重新设计
        self.embed_size = 512  # 适中的嵌入维度，避免过拟合
        self.word_dim = 300
        self.vocab_size = 20000
        self.img_dim = 2048
        
        # 注意力机制配置
        self.num_heads = 8  # 多头注意力
        self.num_layers = 2  # 层数
        self.dropout = 0.1
        
        # 训练配置 - 更保守的设置
        self.batch_size = 16  # 减小批次，确保训练稳定
        self.num_epochs = 50
        self.lr_vse = 0.0002  # 更小的学习率
        self.lr_warmup = 0.00001  # 预热学习率
        self.workers = 2
        
        # 损失配置 - 重新设计
        self.margin = 0.2
        self.temperature = 0.07  # InfoNCE温度
        self.lambda_diversity = 0.1  # 多样性损失权重
        self.lambda_contrastive = 1.0
        self.lambda_triplet = 0.5
        
        # 训练策略
        self.warmup_epochs = 3
        self.cosine_lr = True  # 余弦退火
        self.grad_clip = 2.0  # 梯度裁剪
        
        # 其他配置
        self.log_step = 50
        self.val_step = 200
        self.logger_name = './runs/tk_optimized/log'
        self.model_name = './runs/tk_optimized/checkpoint'
        
        # 早停配置
        self.early_stop_patience = 10
        self.target_performance = 70.0  # 目标R@1


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.out = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_size
        )
        
        output = self.out(context)
        return output, attn_weights


class OptimizedImageEncoder(nn.Module):
    """优化的图像编码器 - 加入注意力机制"""
    
    def __init__(self, img_dim, embed_size, num_heads=8, dropout=0.1):
        super(OptimizedImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 区域特征投影
        self.fc_regions = nn.Linear(img_dim, embed_size)
        
        # 多头自注意力
        self.self_attention = MultiHeadAttention(embed_size, num_heads, dropout)
        
        # Layer Norm和MLP
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size)
        )
        
        # 全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images):
        """前向传播"""
        batch_size = images.size(0)
        
        # 处理区域特征
        if len(images.shape) == 3:  # (batch, regions, features)
            regions = self.fc_regions(images)  # (batch, regions, embed_size)
            
            # 自注意力
            attended, _ = self.self_attention(regions, regions, regions)
            regions = self.layer_norm1(regions + attended)
            
            # MLP
            mlp_out = self.mlp(regions)
            regions = self.layer_norm2(regions + mlp_out)
            
            # 全局平均池化
            features = regions.mean(dim=1)  # (batch, embed_size)
        else:
            features = self.fc_regions(images)
        
        features = self.dropout(features)
        
        # L2标准化
        features = F.normalize(features, p=2, dim=1)
        
        return features


class OptimizedTextEncoder(nn.Module):
    """优化的文本编码器 - LSTM + 注意力"""
    
    def __init__(self, vocab_size, word_dim, embed_size, num_layers=2, dropout=0.1):
        super(OptimizedTextEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 词嵌入
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            word_dim, embed_size // 2, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        # 词嵌入初始化
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        
        # LSTM初始化
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, captions, lengths):
        """前向传播"""
        # 词嵌入
        embedded = self.embed(captions)
        embedded = self.embed_dropout(embedded)
        
        # 打包序列
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )
        
        # 注意力权重
        attn_weights = self.attention(lstm_out)  # (batch, max_len, 1)
        
        # 创建mask
        mask = torch.zeros(lstm_out.size(0), lstm_out.size(1))
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        if lstm_out.is_cuda:
            mask = mask.cuda()
        
        # 应用mask
        attn_weights = attn_weights.squeeze(2)  # (batch, max_len)
        attn_weights.masked_fill_(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权求和
        features = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        features = self.dropout(features)
        
        # L2标准化
        features = F.normalize(features, p=2, dim=1)
        
        return features


class InfoNCELoss(nn.Module):
    """InfoNCE损失 - 更强的对比学习"""
    
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, img_emb, cap_emb):
        batch_size = img_emb.size(0)
        
        # L2标准化
        img_emb = F.normalize(img_emb, p=2, dim=1)
        cap_emb = F.normalize(cap_emb, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(img_emb, cap_emb.t()) / self.temperature
        
        # 正样本标签
        labels = torch.arange(batch_size)
        if img_emb.is_cuda:
            labels = labels.cuda()
        
        # Image to Text损失
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        
        # Text to Image损失  
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2


class HardNegativeTripletLoss(nn.Module):
    """硬负例三元组损失"""
    
    def __init__(self, margin=0.2, hardest_ratio=0.5):
        super(HardNegativeTripletLoss, self).__init__()
        self.margin = margin
        self.hardest_ratio = hardest_ratio
        
    def forward(self, img_emb, cap_emb):
        batch_size = img_emb.size(0)
        
        # 计算相似度矩阵
        scores = torch.mm(img_emb, cap_emb.t())
        diagonal = scores.diag().view(-1, 1)
        
        # Image to text
        d1 = diagonal.expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        
        # Text to image
        d2 = diagonal.t().expand_as(scores)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        
        # 清除对角线
        mask = torch.eye(batch_size) > 0.5
        if scores.is_cuda:
            mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        
        # 硬负例挖掘
        num_hard = max(1, int(batch_size * self.hardest_ratio))
        
        # 选择最难的负样本
        cost_s_hard, _ = cost_s.topk(num_hard, dim=1)
        cost_im_hard, _ = cost_im.topk(num_hard, dim=0)
        
        return cost_s_hard.mean() + cost_im_hard.mean()


class OptimizedVSEModel(nn.Module):
    """优化的VSE模型"""
    
    def __init__(self, opt):
        super(OptimizedVSEModel, self).__init__()
        
        # 优化的编码器
        self.img_enc = OptimizedImageEncoder(
            img_dim=opt.img_dim,
            embed_size=opt.embed_size,
            num_heads=opt.num_heads,
            dropout=opt.dropout
        )
        
        self.txt_enc = OptimizedTextEncoder(
            vocab_size=opt.vocab_size,
            word_dim=opt.word_dim,
            embed_size=opt.embed_size,
            num_layers=opt.num_layers,
            dropout=opt.dropout
        )
        
        # 损失函数
        self.infonce_loss = InfoNCELoss(temperature=opt.temperature)
        self.triplet_loss = HardNegativeTripletLoss(
            margin=opt.margin, hardest_ratio=0.5
        )
    
    def forward_emb(self, images, captions, lengths):
        """前向传播得到嵌入"""
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb
    
    def forward_loss(self, img_emb, cap_emb):
        """计算综合损失"""
        loss_infonce = self.infonce_loss(img_emb, cap_emb)
        loss_triplet = self.triplet_loss(img_emb, cap_emb)
        
        # 组合损失
        total_loss = loss_infonce + 0.5 * loss_triplet
        
        return total_loss, loss_infonce, loss_triplet


class OptimizedTrainer:
    """优化的训练器"""
    
    def __init__(self):
        self.opt = OptimizedConfig()
        
        # 加载词汇表
        self.vocab = self.load_vocab()
        if self.vocab:
            self.opt.vocab_size = len(self.vocab)
        
        print(f"词汇表大小: {self.opt.vocab_size}")
    
    def load_vocab(self):
        """加载词汇表"""
        try:
            import pickle
            vocab_path = f'./vocab/{self.opt.data_name}_vocab.pkl'
            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"词汇表文件 {vocab_path} 不存在")
                return None
        except Exception as e:
            print(f"加载词汇表失败: {e}")
            return None
    
    def get_learning_rate(self, epoch, step, total_steps):
        """获取学习率 - 预热 + 余弦退火"""
        if epoch < self.opt.warmup_epochs:
            # 预热阶段
            warmup_progress = (epoch * total_steps + step) / (self.opt.warmup_epochs * total_steps)
            return self.opt.lr_warmup + (self.opt.lr_vse - self.opt.lr_warmup) * warmup_progress
        else:
            # 余弦退火
            remaining_epochs = self.opt.num_epochs - self.opt.warmup_epochs
            current_epoch = epoch - self.opt.warmup_epochs
            progress = current_epoch / remaining_epochs
            return self.opt.lr_vse * 0.5 * (1 + math.cos(math.pi * progress))
    
    def train_epoch(self, model, data_loader, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()
        
        end = time.time()
        total_steps = len(data_loader)
        
        for i, train_data in enumerate(data_loader):
            data_time.update(time.time() - end)
            
            # 数据准备
            images, captions, lengths, ids = train_data
            
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            # 动态学习率
            lr = self.get_learning_rate(epoch, i, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 前向传播
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)
            
            # 计算损失
            total_loss, loss_infonce, loss_triplet = model.forward_loss(img_emb, cap_emb)
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)
            
            optimizer.step()
            
            # 记录
            train_logger.update('Loss', total_loss.item(), img_emb.size(0))
            train_logger.update('InfoNCE', loss_infonce.item(), img_emb.size(0))
            train_logger.update('Triplet', loss_triplet.item(), img_emb.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.opt.log_step == 0:
                print(f'Epoch [{epoch+1}][{i}/{total_steps}] '
                      f'LR: {lr:.6f} '
                      f'Time {batch_time.val:.3f} '
                      f'Loss {train_logger.meters["Loss"].val:.4f} '
                      f'InfoNCE {train_logger.meters["InfoNCE"].val:.4f} '
                      f'Triplet {train_logger.meters["Triplet"].val:.4f}')
    
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
        print("=== 优化训练开始 - 目标R@1 >= 70% ===")
        print(f"之前最佳R-sum: 9.37 (需要大幅提升)")
        print(f"数据路径: {self.opt.data_path}")
        print(f"批次大小: {self.opt.batch_size}")
        print(f"嵌入维度: {self.opt.embed_size}")
        print(f"学习率: {self.opt.lr_vse}")
        print(f"目标性能: R@1 >= {self.opt.target_performance}%")
        
        # 创建保存目录
        os.makedirs(self.opt.logger_name, exist_ok=True)
        os.makedirs(self.opt.model_name, exist_ok=True)
        
        # 数据加载器
        if not self.vocab:
            print("词汇表不存在，无法继续训练")
            return
        
        train_loader = get_precomp_loader(
            self.opt.data_path, 'train', self.vocab, self.opt,
            batch_size=self.opt.batch_size, shuffle=True, 
            num_workers=self.opt.workers
        )
        
        val_loader = get_precomp_loader(
            self.opt.data_path, 'dev', self.vocab, self.opt,
            batch_size=self.opt.batch_size, shuffle=False,
            num_workers=self.opt.workers
        )
        
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        
        # 模型
        model = OptimizedVSEModel(self.opt)
        
        if torch.cuda.is_available():
            model.cuda()
            print("使用GPU训练")
        else:
            print("使用CPU训练")
        
        # 优化器 - 使用AdamW
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.opt.lr_vse,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
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
            if rsum > best_rsum:  # 使用rsum作为主要指标
                best_rsum = rsum
                best_avg_r1 = avg_r1
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
                
                print(f"💾 保存最佳模型 - R-sum: {best_rsum:.2f}, 平均R@1: {best_avg_r1:.2f}%")
                
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
                
            print(f"当前最佳 - R-sum: {best_rsum:.2f}, 平均R@1: {best_avg_r1:.2f}% (目标: {self.opt.target_performance}%)")
        
        # 训练总结
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"之前最佳R-sum: 9.37")
        print(f"本次最佳R-sum: {best_rsum:.2f}")
        print(f"提升: {best_rsum - 9.37:.2f}")
        print(f"最佳平均R@1: {best_avg_r1:.2f}%")
        
        if best_avg_r1 >= self.opt.target_performance:
            print("✅ 成功达到70%目标!")
        else:
            improvement_needed = self.opt.target_performance - best_avg_r1
            print(f"距离70%目标还差: {improvement_needed:.2f}%")
            
            if best_rsum > 50:
                print("✅ 模型性能显著提升，继续优化可达到目标")
            elif best_rsum > 20:
                print("⚡ 模型性能有一定提升，需要进一步优化")
            else:
                print("❌ 性能提升不明显，需要重新检查数据和模型")
        
        print(f"{'='*60}")


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = OptimizedTrainer()
    trainer.train()


if __name__ == '__main__':
    main() 