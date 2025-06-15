#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版训练脚本 - 基于37.47%的成功基础冲击70%目标
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

from data_chinese import get_precomp_loader
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector
import vocab


class EnhancedConfig:
    """增强配置 - 基于成功经验进一步优化"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 模型配置 - 适度增强，避免过拟合
        self.embed_size = 768      # 增大但不过度
        self.word_dim = 384        # 适中的词嵌入
        self.vocab_size = 20000
        self.img_dim = 2048
        
        # 网络结构
        self.num_heads = 12        # 适中的注意力头数
        self.num_layers = 3        # 适中层数
        self.dropout = 0.1         # 降低dropout
        self.hidden_dim = 1536     # 适中隐藏层
        
        # 训练配置 - 基于成功经验优化
        self.batch_size = 32       # 稍大批次
        self.num_epochs = 100      # 更多轮次
        self.lr_vse = 8e-5         # 更小学习率
        self.workers = 2
        
        # 损失配置 - 微调参数
        self.margin = 0.1          # 更小边距
        self.temperature = 0.04    # 更低温度
        self.lambda_triplet = 0.8
        self.lambda_infonce = 1.0
        
        # 学习率调度
        self.warmup_epochs = 3
        self.cosine_epochs = 25
        self.grad_clip = 0.5
        
        # 其他配置
        self.log_step = 50
        self.val_step = 200
        self.logger_name = './runs/tk_enhanced/log'
        self.model_name = './runs/tk_enhanced/checkpoint'
        self.early_stop_patience = 20


class EnhancedImageEncoder(nn.Module):
    """增强图像编码器"""
    
    def __init__(self, img_dim, embed_size, num_heads=12, dropout=0.1):
        super(EnhancedImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 图像特征投影
        self.fc = nn.Linear(img_dim, embed_size)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(36, embed_size))
        
        # 多层自注意力
        self.attention1 = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, 
            dropout=dropout, batch_first=True
        )
        self.attention2 = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, 
            dropout=dropout, batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size),
            nn.Dropout(dropout)
        )
        
        # 全局特征池化
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
        batch_size = images.size(0)
        
        if len(images.shape) == 3:  # (batch, regions, features)
            # 特征投影
            x = self.fc(images)  # (batch, 36, embed_size)
            
            # 添加位置编码
            x = x + self.pos_embedding.unsqueeze(0)
            x = self.dropout(x)
            
            # 第一层自注意力
            attn_out1, _ = self.attention1(x, x, x)
            x = self.norm1(x + attn_out1)
            
            # 第二层自注意力
            attn_out2, _ = self.attention2(x, x, x)
            x = self.norm2(x + attn_out2)
            
            # 前馈网络
            ff_out = self.feed_forward(x)
            x = x + ff_out
            
            # 全局池化
            features = x.mean(dim=1)  # (batch, embed_size)
        else:
            features = self.fc(images)
        
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class EnhancedTextEncoder(nn.Module):
    """增强文本编码器"""
    
    def __init__(self, vocab_size, word_dim, embed_size, dropout=0.1):
        super(EnhancedTextEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_dim = word_dim
        
        # 词嵌入
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 双向LSTM - 两层
        self.lstm = nn.LSTM(
            word_dim, embed_size // 2, 2,
            batch_first=True, bidirectional=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=12,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力权重计算
        self.attention_fc = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.Tanh(),
            nn.Linear(embed_size // 2, 1)
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )
        
        self.layer_norm = nn.LayerNorm(embed_size)
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
        batch_size, max_len = captions.size()
        
        # 词嵌入
        embedded = self.embed(captions)
        embedded = self.embed_dropout(embedded)
        
        # LSTM编码
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # 自注意力
        attn_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            attn_mask[i, length:] = True
        
        if lstm_out.is_cuda:
            attn_mask = attn_mask.cuda()
        
        attn_out, _ = self.self_attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=attn_mask
        )
        
        # 残差连接
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # 注意力权重
        attention_weights = self.attention_fc(lstm_out).squeeze(-1)
        
        # 应用长度mask
        mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        if lstm_out.is_cuda:
            mask = mask.cuda()
        
        attention_weights.masked_fill_(mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        features = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # 输出投影
        features = self.output_projection(features)
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class EnhancedVSEModel(nn.Module):
    """增强VSE模型"""
    
    def __init__(self, opt):
        super(EnhancedVSEModel, self).__init__()
        
        self.img_enc = EnhancedImageEncoder(
            img_dim=opt.img_dim,
            embed_size=opt.embed_size,
            num_heads=opt.num_heads,
            dropout=opt.dropout
        )
        
        self.txt_enc = EnhancedTextEncoder(
            vocab_size=opt.vocab_size,
            word_dim=opt.word_dim,
            embed_size=opt.embed_size,
            dropout=opt.dropout
        )
        
        self.margin = opt.margin
        self.temperature = opt.temperature
        self.lambda_triplet = opt.lambda_triplet
        self.lambda_infonce = opt.lambda_infonce
    
    def forward_emb(self, images, captions, lengths):
        """前向传播得到嵌入"""
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb
    
    def forward_loss(self, img_emb, cap_emb):
        """计算损失"""
        batch_size = img_emb.size(0)
        
        # InfoNCE损失
        sim_matrix = torch.mm(img_emb, cap_emb.t()) / self.temperature
        labels = torch.arange(batch_size)
        if img_emb.is_cuda:
            labels = labels.cuda()
        
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
        loss_infonce = (loss_i2t + loss_t2i) / 2
        
        # 三元组损失
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
        
        loss_triplet = cost_s.mean() + cost_im.mean()
        
        # 总损失
        total_loss = (self.lambda_infonce * loss_infonce + 
                     self.lambda_triplet * loss_triplet)
        
        return total_loss, loss_infonce, loss_triplet


class EnhancedTrainer:
    """增强训练器"""
    
    def __init__(self):
        self.opt = EnhancedConfig()
        
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
    
    def adjust_learning_rate(self, optimizer, epoch):
        """调整学习率"""
        if epoch < self.opt.warmup_epochs:
            # 预热阶段
            lr = self.opt.lr_vse * (epoch + 1) / self.opt.warmup_epochs
        else:
            # 余弦退火
            adjusted_epoch = epoch - self.opt.warmup_epochs
            total_cosine_epochs = self.opt.cosine_epochs
            
            if adjusted_epoch < total_cosine_epochs:
                lr = self.opt.lr_vse * 0.5 * (1 + math.cos(math.pi * adjusted_epoch / total_cosine_epochs))
            else:
                # 指数衰减
                decay_epochs = adjusted_epoch - total_cosine_epochs
                lr = self.opt.lr_vse * 0.1 * (0.95 ** decay_epochs)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def train_epoch(self, model, data_loader, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()
        
        end = time.time()
        
        for i, train_data in enumerate(data_loader):
            data_time.update(time.time() - end)
            
            # 数据准备
            images, captions, lengths, ids = train_data
            
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            # 调整学习率
            lr = self.adjust_learning_rate(optimizer, epoch)
            
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
                print(f'Epoch [{epoch+1}][{i}/{len(data_loader)}] '
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
        
        print(f"Text to Image: R@1: {r1i:.2f}%, R@5: {r5i:.2f}%, R@10: {r10i:.2f}%")
        print(f"Image to Text: R@1: {r1:.2f}%, R@5: {r5:.2f}%, R@10: {r10:.2f}%")
        print(f"平均 R@1: {avg_r1:.2f}%")
        print(f"R-sum: {rsum:.2f}")
        
        return rsum, avg_r1
    
    def train(self):
        """主训练流程"""
        print("=== 增强版训练开始 - 冲击70% R@1目标 ===")
        print(f"基于成功经验: R@1 = 37.47%")
        print(f"目标: R@1 >= 70%")
        print(f"优化策略: 增大模型容量+精细学习率调度+更强网络结构")
        
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
        model = EnhancedVSEModel(self.opt)
        
        if torch.cuda.is_available():
            model.cuda()
            print("使用GPU训练")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")
        
        # 优化器
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.opt.lr_vse,
            weight_decay=0.01
        )
        
        # 训练状态
        best_rsum = 0
        best_avg_r1 = 0
        patience_counter = 0
        
        print(f"\n开始训练 {self.opt.num_epochs} epochs...")
        
        for epoch in range(self.opt.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.opt.num_epochs}")
            print("-" * 60)
            
            # 训练
            self.train_epoch(model, train_loader, optimizer, epoch)
            
            # 验证
            rsum, avg_r1 = self.validate(model, val_loader)
            
            # 保存最佳模型
            is_best = avg_r1 > best_avg_r1
            if is_best:
                best_rsum = rsum
                best_avg_r1 = avg_r1
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_rsum': best_rsum,
                    'best_avg_r1': best_avg_r1,
                    'opt': self.opt,
                }
                
                model_path = os.path.join(self.opt.model_name, 'best_model.pth')
                torch.save(checkpoint, model_path)
                
                print(f"💾 保存最佳模型 - R-sum: {best_rsum:.2f}, 平均R@1: {best_avg_r1:.2f}%")
                
                # 检查目标达成
                if best_avg_r1 >= 70.0:
                    print(f"🎉 成功达到70%目标! 训练完成!")
                    break
                    
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= self.opt.early_stop_patience:
                print(f"早停触发")
                break
            
            # 进度报告
            improvement = best_avg_r1 - 37.47
            remaining = 70.0 - best_avg_r1
            
            print(f"当前最佳: R@1 = {best_avg_r1:.2f}%")
            print(f"改进: {improvement:.2f}%, 距离目标: {remaining:.2f}%")
        
        # 训练总结
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"基线: R@1 = 37.47%")
        print(f"结果: R@1 = {best_avg_r1:.2f}%")
        print(f"提升: {best_avg_r1 - 37.47:.2f}%")
        
        if best_avg_r1 >= 70.0:
            print("🎉 成功达成70%目标!")
        else:
            print(f"距离目标还差: {70.0 - best_avg_r1:.2f}%")


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = EnhancedTrainer()
    trainer.train()


if __name__ == '__main__':
    main() 