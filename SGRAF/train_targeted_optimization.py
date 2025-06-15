#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对性优化训练脚本 - 基于37.47%成功基础的专门优化

关键洞察：
1. 已知的成功配置：37.47% R@1
2. 数据特点：2910训练样本，1562词汇，36区域特征
3. 优化方向：特征表示增强 + 损失函数改进

策略：
- 更强的特征融合
- 改进的注意力机制
- 标签平滑和正则化
- 渐进式训练策略
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


class Config:
    """配置类"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 模型配置
        self.embed_size = 1024     # 大特征维度
        self.word_dim = 384        # 词嵌入维度
        self.vocab_size = 20000
        self.img_dim = 2048
        
        # 网络配置
        self.num_heads = 8
        self.dropout = 0.25        # 适度dropout
        
        # 训练配置
        self.batch_size = 20       # 平衡批次大小
        self.num_epochs = 120      # 充分训练
        self.lr_vse = 8e-5         # 稳定学习率
        self.workers = 2
        
        # 损失配置
        self.margin = 0.08
        self.temperature = 0.04
        self.lambda_triplet = 0.6
        self.lambda_infonce = 1.0
        
        # 训练策略
        self.warmup_epochs = 3
        self.cosine_epochs = 30
        self.grad_clip = 0.8
        self.weight_decay = 0.015
        
        # 其他配置
        self.log_step = 25
        self.logger_name = './runs/tk_targeted/log'
        self.model_name = './runs/tk_targeted/checkpoint'
        self.early_stop_patience = 20


class EnhancedImageEncoder(nn.Module):
    """增强图像编码器"""
    
    def __init__(self, img_dim, embed_size, num_heads=8, dropout=0.25):
        super(EnhancedImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 特征映射
        self.feature_proj = nn.Sequential(
            nn.Linear(img_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(36, embed_size) * 0.02)
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size),
            nn.Dropout(dropout)
        )
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.Tanh(),
            nn.Linear(embed_size // 2, 1)
        )
        
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images):
        batch_size = images.size(0)
        
        if len(images.shape) == 3:  # (batch, 36, 2048)
            # 特征投影
            x = self.feature_proj(images)  # (batch, 36, embed_size)
            
            # 位置编码
            x = x + self.pos_embedding.unsqueeze(0)
            x = self.dropout(x)
            
            # 自注意力
            attn_out, _ = self.self_attention(x, x, x)
            x = self.layer_norm1(x + attn_out)
            
            # 前馈网络
            ff_out = self.feed_forward(x)
            x = self.layer_norm2(x + ff_out)
            
            # 注意力池化
            attention_weights = self.attention_pool(x).squeeze(-1)  # (batch, 36)
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # 加权聚合
            features = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        else:
            features = self.feature_proj(images)
        
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class EnhancedTextEncoder(nn.Module):
    """增强文本编码器"""
    
    def __init__(self, vocab_size, word_dim, embed_size, dropout=0.25):
        super(EnhancedTextEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_dim = word_dim
        
        # 词嵌入
        self.embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        # LSTM
        self.lstm = nn.LSTM(
            word_dim, embed_size // 2, 2,
            batch_first=True, bidirectional=True,
            dropout=dropout
        )
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力机制
        self.attention_fc = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, 1)
        )
        
        # 特征增强
        self.enhance_fc = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )
        
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        
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
        
        # LSTM
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
        
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # 注意力权重
        attention_weights = self.attention_fc(lstm_out).squeeze(-1)
        
        # 长度mask
        mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        if lstm_out.is_cuda:
            mask = mask.cuda()
        
        attention_weights.masked_fill_(mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        features = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # 特征增强
        features = self.enhance_fc(features)
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class VSEModel(nn.Module):
    """VSE模型"""
    
    def __init__(self, opt):
        super(VSEModel, self).__init__()
        
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
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb
    
    def forward_loss(self, img_emb, cap_emb):
        batch_size = img_emb.size(0)
        
        # InfoNCE损失
        sim_matrix = torch.mm(img_emb, cap_emb.t()) / self.temperature
        labels = torch.arange(batch_size)
        if img_emb.is_cuda:
            labels = labels.cuda()
        
        loss_i2t = F.cross_entropy(sim_matrix, labels, label_smoothing=0.05)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels, label_smoothing=0.05)
        loss_infonce = (loss_i2t + loss_t2i) / 2
        
        # 三元组损失
        scores = torch.mm(img_emb, cap_emb.t())
        diagonal = scores.diag().view(-1, 1)
        
        d1 = diagonal.expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        
        d2 = diagonal.t().expand_as(scores)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        
        mask = torch.eye(batch_size) > 0.5
        if scores.is_cuda:
            mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        
        loss_triplet = cost_s.mean() + cost_im.mean()
        
        total_loss = (self.lambda_infonce * loss_infonce + 
                     self.lambda_triplet * loss_triplet)
        
        return total_loss, loss_infonce, loss_triplet


class Trainer:
    """训练器"""
    
    def __init__(self):
        self.opt = Config()
        
        self.vocab = self.load_vocab()
        if self.vocab:
            self.opt.vocab_size = len(self.vocab)
        
        print(f"词汇表大小: {self.opt.vocab_size}")
    
    def load_vocab(self):
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
        if epoch < self.opt.warmup_epochs:
            lr = self.opt.lr_vse * (epoch + 1) / self.opt.warmup_epochs
        else:
            adjusted_epoch = epoch - self.opt.warmup_epochs
            
            if adjusted_epoch < self.opt.cosine_epochs:
                lr = self.opt.lr_vse * 0.5 * (1 + math.cos(math.pi * adjusted_epoch / self.opt.cosine_epochs))
            else:
                decay_epochs = adjusted_epoch - self.opt.cosine_epochs
                lr = self.opt.lr_vse * 0.2 * (0.96 ** decay_epochs)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def train_epoch(self, model, data_loader, optimizer, epoch):
        model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_logger = LogCollector()
        
        end = time.time()
        
        for i, train_data in enumerate(data_loader):
            data_time.update(time.time() - end)
            
            images, captions, lengths, ids = train_data
            
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            lr = self.adjust_learning_rate(optimizer, epoch)
            
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)
            total_loss, loss_infonce, loss_triplet = model.forward_loss(img_emb, cap_emb)
            
            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)
            optimizer.step()
            
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
        print("=== 针对性优化训练 ===")
        print(f"基线性能: R@1 = 37.47%")
        print(f"优化目标: 进一步提升到45%+")
        
        os.makedirs(self.opt.logger_name, exist_ok=True)
        os.makedirs(self.opt.model_name, exist_ok=True)
        
        if not self.vocab:
            print("词汇表不存在，退出")
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
        
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
        
        model = VSEModel(self.opt)
        
        if torch.cuda.is_available():
            model.cuda()
            print("使用GPU")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params:,}")
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.opt.lr_vse,
            weight_decay=self.opt.weight_decay
        )
        
        best_rsum = 0
        best_avg_r1 = 0
        patience_counter = 0
        
        print(f"\n开始训练 {self.opt.num_epochs} epochs...")
        
        for epoch in range(self.opt.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.opt.num_epochs}")
            print("-" * 50)
            
            self.train_epoch(model, train_loader, optimizer, epoch)
            rsum, avg_r1 = self.validate(model, val_loader)
            
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
                
                print(f"💾 最佳模型 - R-sum: {best_rsum:.2f}, 平均R@1: {best_avg_r1:.2f}%")
                
                if best_avg_r1 >= 45.0:
                    print(f"🎉 达到45%目标!")
                if best_avg_r1 >= 70.0:
                    print(f"🎉🎉🎉 达到70%最终目标!")
                    break
                    
            else:
                patience_counter += 1
                
            if patience_counter >= self.opt.early_stop_patience:
                print(f"早停")
                break
            
            improvement = best_avg_r1 - 37.47
            remaining = 70.0 - best_avg_r1
            
            print(f"当前最佳: {best_avg_r1:.2f}%")
            print(f"改进: {improvement:.2f}%")
            print(f"距离70%: {remaining:.2f}%")
        
        print(f"\n{'='*50}")
        print("训练完成!")
        print(f"基线: 37.47%")
        print(f"结果: {best_avg_r1:.2f}%")
        print(f"改进: {best_avg_r1 - 37.47:.2f}%")
        
        if best_avg_r1 >= 70.0:
            print("🎉 达到70%目标!")
        elif best_avg_r1 >= 45.0:
            print("⭐ 达到45%中期目标!")
        else:
            print(f"继续努力，距离45%还差: {45.0 - best_avg_r1:.2f}%")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main() 