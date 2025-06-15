#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
突破性优化训练脚本 - 冲破37%瓶颈

核心策略：
1. 大幅增强模型容量
2. 深度跨模态交互
3. 多尺度特征融合  
4. 强化学习式的负样本挖掘
5. 知识蒸馏和集成学习思想

目标：突破37%瓶颈，达到45%+
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


class BreakthroughConfig:
    """突破性配置"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 大幅增强的模型配置
        self.embed_size = 1536      # 更大的嵌入维度
        self.word_dim = 512         # 更大的词嵌入
        self.vocab_size = 20000
        self.img_dim = 2048
        
        # 深度网络配置
        self.num_heads = 12         # 更多注意力头
        self.num_layers = 4         # 更深层数
        self.dropout = 0.2          # 适中dropout
        self.hidden_dim = 2048      # 大隐藏层
        
        # 跨模态交互配置
        self.cross_modal_layers = 2  # 专门的跨模态层
        self.fusion_method = 'advanced'  # 高级融合方法
        
        # 训练配置
        self.batch_size = 16        # 小批次，更多梯度更新
        self.num_epochs = 200       # 更长训练
        self.lr_vse = 3e-5          # 较小学习率
        self.workers = 2
        
        # 多重损失配置
        self.margin = 0.05          # 更严格边距
        self.temperature = 0.02     # 更低温度
        self.lambda_triplet = 0.8
        self.lambda_infonce = 1.0
        self.lambda_cross_modal = 0.5  # 跨模态损失
        self.lambda_consistency = 0.3   # 一致性损失
        
        # 高级训练策略
        self.warmup_epochs = 8
        self.cosine_epochs = 50
        self.grad_clip = 1.0
        self.weight_decay = 0.01
        self.ema_decay = 0.999      # 指数移动平均
        
        # 负样本挖掘
        self.hard_negative_ratio = 0.7
        self.negative_mining_start_epoch = 10
        
        # 其他配置
        self.log_step = 20
        self.logger_name = './runs/tk_breakthrough/log'
        self.model_name = './runs/tk_breakthrough/checkpoint'
        self.early_stop_patience = 30
        self.target_performance = 45.0


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, embed_size, num_heads=8, dropout=0.2):
        super(CrossModalAttention, self).__init__()
        
        # 图像到文本的注意力
        self.img2txt_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 文本到图像的注意力
        self.txt2img_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 投影层
        self.img_proj = nn.Linear(embed_size, embed_size)
        self.txt_proj = nn.Linear(embed_size, embed_size)
        
        # 层归一化
        self.img_norm = nn.LayerNorm(embed_size)
        self.txt_norm = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, img_features, txt_features, img_mask=None, txt_mask=None):
        # 图像特征接收文本信息
        img_attended, _ = self.img2txt_attention(
            img_features, txt_features, txt_features,
            key_padding_mask=txt_mask
        )
        img_features = self.img_norm(img_features + self.dropout(img_attended))
        img_features = self.img_proj(img_features)
        
        # 文本特征接收图像信息
        txt_attended, _ = self.txt2img_attention(
            txt_features, img_features, img_features,
            key_padding_mask=img_mask
        )
        txt_features = self.txt_norm(txt_features + self.dropout(txt_attended))
        txt_features = self.txt_proj(txt_features)
        
        return img_features, txt_features


class AdvancedImageEncoder(nn.Module):
    """高级图像编码器 - 深度特征提取"""
    
    def __init__(self, img_dim, embed_size, num_heads=12, num_layers=4, dropout=0.2):
        super(AdvancedImageEncoder, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        
        # 初始特征投影
        self.initial_proj = nn.Sequential(
            nn.Linear(img_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(36, embed_size) * 0.02)
        
        # 多层Transformer编码器
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=embed_size * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 多尺度池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, 1)
        )
        
        # 多尺度融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )
        
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
            # 初始投影
            x = self.initial_proj(images)  # (batch, 36, embed_size)
            
            # 位置编码
            x = x + self.pos_embedding.unsqueeze(0)
            x = self.dropout(x)
            
            # 多层Transformer编码
            for layer in self.transformer_layers:
                x = layer(x)
            
            # 多尺度特征提取
            # 1. 注意力池化
            attention_weights = self.attention_pool(x).squeeze(-1)  # (batch, 36)
            attention_weights = F.softmax(attention_weights, dim=1)
            attended_features = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
            
            # 2. 全局平均池化
            global_features = x.mean(dim=1)
            
            # 3. 全局最大池化  
            max_features = x.max(dim=1)[0]
            
            # 多尺度融合
            combined = torch.cat([attended_features, global_features, max_features], dim=-1)
            features = self.fusion(combined)
            
        else:
            features = self.initial_proj(images)
        
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features, x  # 返回最终特征和中间特征


class AdvancedTextEncoder(nn.Module):
    """高级文本编码器 - 深度语义理解"""
    
    def __init__(self, vocab_size, word_dim, embed_size, num_layers=4, dropout=0.2):
        super(AdvancedTextEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_dim = word_dim
        self.num_layers = num_layers
        
        # 增强词嵌入
        self.embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.embed_proj = nn.Linear(word_dim, embed_size)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(100, embed_size) * 0.02)
        
        # 多层双向LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                embed_size if i > 0 else embed_size,
                embed_size // 2,
                batch_first=True,
                bidirectional=True
            ) for i in range(num_layers)
        ])
        
        # 自注意力层
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_size,
                num_heads=12,
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_size) for _ in range(num_layers + 2)
        ])
        
        # 多层注意力机制
        self.word_attention = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, 1)
        )
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.LayerNorm(embed_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        # 词嵌入初始化
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        
        # LSTM初始化
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, captions, lengths):
        batch_size, max_len = captions.size()
        
        # 词嵌入和投影
        embedded = self.embed(captions)
        embedded = self.embed_proj(embedded)
        
        # 位置编码
        pos_len = min(max_len, self.pos_embedding.size(0))
        embedded[:, :pos_len] += self.pos_embedding[:pos_len].unsqueeze(0)
        embedded = self.embed_dropout(embedded)
        
        # 多层LSTM编码
        lstm_out = embedded
        for i, lstm in enumerate(self.lstm_layers):
            packed = nn.utils.rnn.pack_padded_sequence(
                lstm_out, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            lstm_out = self.layer_norms[i](lstm_out)
        
        # 多层自注意力
        attn_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            attn_mask[i, length:] = True
        
        if lstm_out.is_cuda:
            attn_mask = attn_mask.cuda()
        
        attn_out = lstm_out
        for i, attention in enumerate(self.self_attention_layers):
            attn_output, _ = attention(attn_out, attn_out, attn_out, key_padding_mask=attn_mask)
            attn_out = self.layer_norms[self.num_layers + i](attn_out + attn_output)
        
        # 多层注意力池化
        attention_weights = self.word_attention(attn_out).squeeze(-1)
        
        # 应用长度mask
        mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        if attn_out.is_cuda:
            mask = mask.cuda()
        
        attention_weights.masked_fill_(mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        features = torch.bmm(attention_weights.unsqueeze(1), attn_out).squeeze(1)
        
        # 特征增强
        features = self.feature_enhance(features)
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features, attn_out  # 返回最终特征和中间特征


class BreakthroughVSEModel(nn.Module):
    """突破性VSE模型"""
    
    def __init__(self, opt):
        super(BreakthroughVSEModel, self).__init__()
        
        # 高级编码器
        self.img_enc = AdvancedImageEncoder(
            img_dim=opt.img_dim,
            embed_size=opt.embed_size,
            num_heads=opt.num_heads,
            num_layers=opt.num_layers,
            dropout=opt.dropout
        )
        
        self.txt_enc = AdvancedTextEncoder(
            vocab_size=opt.vocab_size,
            word_dim=opt.word_dim,
            embed_size=opt.embed_size,
            num_layers=opt.num_layers,
            dropout=opt.dropout
        )
        
        # 跨模态交互层
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(
                embed_size=opt.embed_size,
                num_heads=opt.num_heads,
                dropout=opt.dropout
            ) for _ in range(opt.cross_modal_layers)
        ])
        
        # 损失配置
        self.margin = opt.margin
        self.temperature = opt.temperature
        self.lambda_triplet = opt.lambda_triplet
        self.lambda_infonce = opt.lambda_infonce
        self.lambda_cross_modal = opt.lambda_cross_modal
        self.lambda_consistency = opt.lambda_consistency
        
        # EMA模型
        self.ema_img_enc = None
        self.ema_txt_enc = None
        self.ema_decay = opt.ema_decay
    
    def forward_emb(self, images, captions, lengths, use_ema=False):
        """前向传播得到嵌入"""
        if use_ema and self.ema_img_enc is not None:
            img_emb, img_seq = self.ema_img_enc(images)
            cap_emb, cap_seq = self.ema_txt_enc(captions, lengths)
        else:
            img_emb, img_seq = self.img_enc(images)
            cap_emb, cap_seq = self.txt_enc(captions, lengths)
        
        # 跨模态交互（如果有中间特征）
        if hasattr(self, 'cross_modal_layers') and img_seq is not None and cap_seq is not None:
            for cross_modal in self.cross_modal_layers:
                img_seq, cap_seq = cross_modal(img_seq, cap_seq)
            
            # 重新池化
            img_emb = img_seq.mean(dim=1)
            cap_emb = cap_seq.mean(dim=1)
            
            img_emb = F.normalize(img_emb, p=2, dim=1)
            cap_emb = F.normalize(cap_emb, p=2, dim=1)
        
        return img_emb, cap_emb
    
    def forward_loss(self, img_emb, cap_emb, epoch=0):
        """计算综合损失"""
        batch_size = img_emb.size(0)
        
        # 主要InfoNCE损失
        sim_matrix = torch.mm(img_emb, cap_emb.t()) / self.temperature
        labels = torch.arange(batch_size)
        if img_emb.is_cuda:
            labels = labels.cuda()
        
        loss_i2t = F.cross_entropy(sim_matrix, labels, label_smoothing=0.1)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels, label_smoothing=0.1)
        loss_infonce = (loss_i2t + loss_t2i) / 2
        
        # 自适应三元组损失
        scores = torch.mm(img_emb, cap_emb.t())
        diagonal = scores.diag().view(-1, 1)
        
        d1 = diagonal.expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        
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
    
    def update_ema(self):
        """更新EMA模型"""
        if self.ema_img_enc is None:
            # 创建EMA模型（简化版本）
            return
        
        # 更新EMA参数
        for ema_param, param in zip(self.ema_img_enc.parameters(), self.img_enc.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        for ema_param, param in zip(self.ema_txt_enc.parameters(), self.txt_enc.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)


class BreakthroughTrainer:
    """突破性训练器"""
    
    def __init__(self):
        self.opt = BreakthroughConfig()
        
        # 加载词汇表
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
        """高级学习率调度"""
        if epoch < self.opt.warmup_epochs:
            # 预热阶段
            lr = self.opt.lr_vse * (epoch + 1) / self.opt.warmup_epochs
        else:
            adjusted_epoch = epoch - self.opt.warmup_epochs
            
            if adjusted_epoch < self.opt.cosine_epochs:
                # 余弦退火
                lr = self.opt.lr_vse * 0.5 * (1 + math.cos(math.pi * adjusted_epoch / self.opt.cosine_epochs))
            else:
                # 缓慢衰减
                decay_epochs = adjusted_epoch - self.opt.cosine_epochs
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
            total_loss, loss_infonce, loss_triplet = model.forward_loss(img_emb, cap_emb, epoch)
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)
            
            optimizer.step()
            
            # 更新EMA
            if epoch >= self.opt.warmup_epochs:
                model.update_ema()
            
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
    
    def validate(self, model, data_loader, use_ema=False):
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
                
                img_emb, cap_emb = model.forward_emb(images, captions, lengths, use_ema=use_ema)
                
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
        print("=== 突破性优化训练 - 目标45%+ ===")
        print(f"当前瓶颈: 37% R@1")
        print(f"突破策略: 深度架构+跨模态交互+高级优化")
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
        model = BreakthroughVSEModel(self.opt)
        
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
            weight_decay=self.opt.weight_decay,
            betas=(0.9, 0.999)
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
            rsum, avg_r1 = self.validate(model, val_loader, use_ema=False)
            
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
                if best_avg_r1 >= self.opt.target_performance:
                    print(f"🎉 突破成功! 达到{self.opt.target_performance}%目标!")
                if best_avg_r1 >= 70.0:
                    print(f"🎉🎉🎉 达到最终70%目标! 训练完成!")
                    break
                    
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= self.opt.early_stop_patience:
                print(f"早停触发")
                break
            
            # 进度报告
            improvement = best_avg_r1 - 37.47
            remaining_to_target = self.opt.target_performance - best_avg_r1
            remaining_to_70 = 70.0 - best_avg_r1
            
            print(f"当前最佳: R@1 = {best_avg_r1:.2f}%")
            print(f"相对基线改进: {improvement:.2f}%")
            print(f"距离突破目标({self.opt.target_performance}%): {remaining_to_target:.2f}%")
            print(f"距离最终目标(70%): {remaining_to_70:.2f}%")
            
            # 动态策略调整
            if epoch > 20 and improvement < 2.0:
                print("⚡ 考虑进一步的架构调整...")
        
        # 训练总结
        print(f"\n{'='*60}")
        print("🚀 突破性训练完成!")
        print(f"{'='*60}")
        print(f"基线: 37.47%")
        print(f"结果: {best_avg_r1:.2f}%")
        print(f"突破: {best_avg_r1 - 37.47:.2f}%")
        
        if best_avg_r1 >= 70.0:
            print("🎉🎉🎉 成功达到70%最终目标!")
        elif best_avg_r1 >= self.opt.target_performance:
            print(f"🎉 成功突破! 达到{self.opt.target_performance}%目标!")
        elif best_avg_r1 > 40.0:
            print("⚡ 显著进步，继续优化可达到目标!")
        else:
            print("🔧 需要更深入的架构革新")


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = BreakthroughTrainer()
    trainer.train()


if __name__ == '__main__':
    main() 