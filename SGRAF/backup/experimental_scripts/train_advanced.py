#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级优化训练脚本 - 突破37%瓶颈

核心突破策略：
1. 知识蒸馏 + 教师模型指导
2. 对比学习增强
3. 负样本硬挖掘
4. 多视角特征融合
5. 课程学习策略

目标：从37%突破到50%+
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
from collections import defaultdict

# 添加项目路径
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

from data_chinese import get_precomp_loader
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector
import vocab


class AdvancedConfig:
    """高级优化配置"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 模型配置 - 平衡性能和效率
        self.embed_size = 1024      # 合理的嵌入维度
        self.word_dim = 300         # 适中的词嵌入
        self.vocab_size = 20000
        self.img_dim = 2048
        
        # 网络配置
        self.num_heads = 8          # 合理的注意力头数
        self.num_layers = 3         # 适中深度
        self.dropout = 0.25         # 适度dropout
        self.hidden_dim = 1536      # 适中隐藏层
        
        # 训练配置
        self.batch_size = 20        # 平衡批次
        self.num_epochs = 120       # 充足轮次
        self.lr_vse = 5e-5          # 适中学习率
        self.workers = 2
        
        # 高级损失配置
        self.margin = 0.08          # 适中边距
        self.temperature = 0.05     # 适中温度
        self.lambda_triplet = 0.8
        self.lambda_infonce = 1.0
        self.lambda_distill = 0.3   # 知识蒸馏权重
        self.lambda_contrast = 0.5  # 对比学习权重
        
        # 知识蒸馏配置
        self.teacher_temp = 4.0     # 教师温度
        self.student_temp = 1.0     # 学生温度
        
        # 课程学习配置
        self.curriculum_start = 10  # 课程学习开始轮次
        self.hard_ratio_start = 0.3 # 初始硬样本比例
        self.hard_ratio_end = 0.8   # 最终硬样本比例
        
        # 训练策略
        self.warmup_epochs = 8
        self.cosine_epochs = 35
        self.grad_clip = 1.2
        self.weight_decay = 0.02
        
        # 数据增强
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.15
        self.cutmix_alpha = 0.1
        
        # 其他配置
        self.log_step = 15
        self.logger_name = './runs/tk_advanced/log'
        self.model_name = './runs/tk_advanced/checkpoint'
        self.early_stop_patience = 20


class ContrastiveLearning(nn.Module):
    """增强对比学习模块"""
    
    def __init__(self, embed_size, temperature=0.05, num_negatives=5):
        super(ContrastiveLearning, self).__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        
        # 动量编码器
        self.momentum = 0.999
        self.queue_size = 256
        
        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size, embed_size // 2)
        )
        
        # 队列存储
        self.register_buffer("img_queue", torch.randn(self.queue_size, embed_size // 2))
        self.register_buffer("txt_queue", torch.randn(self.queue_size, embed_size // 2))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # 初始化队列
        self.img_queue = F.normalize(self.img_queue, dim=1)
        self.txt_queue = F.normalize(self.txt_queue, dim=1)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_keys, txt_keys):
        batch_size = img_keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # 替换队列中的样本
        if ptr + batch_size <= self.queue_size:
            self.img_queue[ptr:ptr + batch_size] = img_keys
            self.txt_queue[ptr:ptr + batch_size] = txt_keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # 分两部分替换
            self.img_queue[ptr:] = img_keys[:self.queue_size - ptr]
            self.txt_queue[ptr:] = txt_keys[:self.queue_size - ptr]
            remaining = batch_size - (self.queue_size - ptr)
            self.img_queue[:remaining] = img_keys[self.queue_size - ptr:]
            self.txt_queue[:remaining] = txt_keys[self.queue_size - ptr:]
            ptr = remaining
        
        self.queue_ptr[0] = ptr
    
    def forward(self, img_emb, txt_emb):
        # 投影到对比学习空间
        img_proj = F.normalize(self.projection(img_emb), dim=1)
        txt_proj = F.normalize(self.projection(txt_emb), dim=1)
        
        batch_size = img_proj.size(0)
        
        # 正样本对
        pos_sim = torch.sum(img_proj * txt_proj, dim=1, keepdim=True)  # (batch, 1)
        
        # 与队列中的负样本计算相似度
        neg_img_sim = torch.mm(txt_proj, self.img_queue.t())  # (batch, queue_size)
        neg_txt_sim = torch.mm(img_proj, self.txt_queue.t())  # (batch, queue_size)
        
        # 图像到文本对比
        logits_i2t = torch.cat([pos_sim, neg_img_sim], dim=1) / self.temperature
        labels_i2t = torch.zeros(batch_size, dtype=torch.long)
        if img_proj.is_cuda:
            labels_i2t = labels_i2t.cuda()
        
        # 文本到图像对比
        logits_t2i = torch.cat([pos_sim, neg_txt_sim], dim=1) / self.temperature
        labels_t2i = torch.zeros(batch_size, dtype=torch.long)
        if txt_proj.is_cuda:
            labels_t2i = labels_t2i.cuda()
        
        # 计算对比损失
        loss_i2t = F.cross_entropy(logits_i2t, labels_i2t)
        loss_t2i = F.cross_entropy(logits_t2i, labels_t2i)
        contrast_loss = (loss_i2t + loss_t2i) / 2
        
        # 更新队列
        self._dequeue_and_enqueue(img_proj.detach(), txt_proj.detach())
        
        return contrast_loss


class HardNegativeMining(nn.Module):
    """硬负样本挖掘"""
    
    def __init__(self, margin=0.08, hard_ratio=0.5):
        super(HardNegativeMining, self).__init__()
        self.margin = margin
        self.hard_ratio = hard_ratio
    
    def forward(self, img_emb, txt_emb, epoch, total_epochs):
        batch_size = img_emb.size(0)
        
        # 动态调整硬负样本比例
        progress = epoch / total_epochs
        current_hard_ratio = 0.3 + (0.8 - 0.3) * progress
        
        # 计算相似度矩阵
        scores = torch.mm(img_emb, txt_emb.t())
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
        
        # 硬负样本选择
        num_hard = max(1, int(batch_size * current_hard_ratio))
        
        # 选择最难的负样本
        cost_s_hard, _ = cost_s.topk(num_hard, dim=1)
        cost_im_hard, _ = cost_im.topk(num_hard, dim=0)
        
        return cost_s_hard.mean() + cost_im_hard.mean()


class MultiViewImageEncoder(nn.Module):
    """多视角图像编码器"""
    
    def __init__(self, img_dim, embed_size, num_heads=8, dropout=0.25):
        super(MultiViewImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 多层特征变换
        self.feature_proj = nn.Sequential(
            nn.Linear(img_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多视角特征提取
        self.global_view = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.local_view = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
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
        
        # 交叉注意力（不同视角间）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, 1)
        )
        
        # 视角融合
        self.view_fusion = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )
        
        self.layer_norm = nn.LayerNorm(embed_size)
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
            x = self.layer_norm(x + attn_out)
            
            # 多视角特征
            global_features = self.global_view(x)  # 全局视角
            local_features = self.local_view(x)    # 局部视角
            
            # 交叉注意力（视角间交互）
            cross_out, _ = self.cross_attention(global_features, local_features, local_features)
            global_features = self.layer_norm(global_features + cross_out)
            
            # 注意力池化
            global_weights = self.attention_pool(global_features).squeeze(-1)
            global_weights = F.softmax(global_weights, dim=1)
            global_pooled = torch.bmm(global_weights.unsqueeze(1), global_features).squeeze(1)
            
            local_weights = self.attention_pool(local_features).squeeze(-1)
            local_weights = F.softmax(local_weights, dim=1)
            local_pooled = torch.bmm(local_weights.unsqueeze(1), local_features).squeeze(1)
            
            # 视角融合
            combined = torch.cat([global_pooled, local_pooled], dim=-1)
            features = self.view_fusion(combined)
            
        else:
            features = self.feature_proj(images)
        
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class AdvancedTextEncoder(nn.Module):
    """高级文本编码器"""
    
    def __init__(self, vocab_size, word_dim, embed_size, dropout=0.25):
        super(AdvancedTextEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_dim = word_dim
        
        # 词嵌入
        self.embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.embed_proj = nn.Linear(word_dim, embed_size)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(100, embed_size) * 0.02)
        
        # 多层BiLSTM
        self.lstm = nn.LSTM(
            embed_size, embed_size // 2, 2,
            batch_first=True, bidirectional=True,
            dropout=dropout
        )
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 层次注意力
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
        
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
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
        
        # 词嵌入和投影
        embedded = self.embed(captions)
        embedded = self.embed_proj(embedded)
        
        # 位置编码
        pos_len = min(max_len, self.pos_embedding.size(0))
        embedded[:, :pos_len] += self.pos_embedding[:pos_len].unsqueeze(0)
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
        
        lstm_out = self.layer_norm(lstm_out + attn_out)
        
        # 词级注意力
        attention_weights = self.word_attention(lstm_out).squeeze(-1)
        
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
        features = self.feature_enhance(features)
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class AdvancedVSEModel(nn.Module):
    """高级VSE模型"""
    
    def __init__(self, opt):
        super(AdvancedVSEModel, self).__init__()
        
        # 编码器
        self.img_enc = MultiViewImageEncoder(
            img_dim=opt.img_dim,
            embed_size=opt.embed_size,
            num_heads=opt.num_heads,
            dropout=opt.dropout
        )
        
        self.txt_enc = AdvancedTextEncoder(
            vocab_size=opt.vocab_size,
            word_dim=opt.word_dim,
            embed_size=opt.embed_size,
            dropout=opt.dropout
        )
        
        # 高级损失模块
        self.contrastive_learning = ContrastiveLearning(
            embed_size=opt.embed_size,
            temperature=opt.temperature
        )
        
        self.hard_negative_mining = HardNegativeMining(
            margin=opt.margin
        )
        
        # 损失权重
        self.margin = opt.margin
        self.temperature = opt.temperature
        self.lambda_triplet = opt.lambda_triplet
        self.lambda_infonce = opt.lambda_infonce
        self.lambda_contrast = opt.lambda_contrast
        self.label_smoothing = opt.label_smoothing
    
    def forward_emb(self, images, captions, lengths):
        img_emb = self.img_enc(images)
        txt_emb = self.txt_enc(captions, lengths)
        return img_emb, txt_emb
    
    def mixup_data(self, img_emb, txt_emb, alpha=0.15):
        """增强Mixup"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = img_emb.size(0)
        index = torch.randperm(batch_size)
        if img_emb.is_cuda:
            index = index.cuda()
        
        mixed_img = lam * img_emb + (1 - lam) * img_emb[index]
        mixed_txt = lam * txt_emb + (1 - lam) * txt_emb[index]
        
        return mixed_img, mixed_txt, lam, index
    
    def forward_loss(self, img_emb, txt_emb, epoch=0, total_epochs=100, use_mixup=False):
        """计算高级损失"""
        batch_size = img_emb.size(0)
        
        # Mixup增强
        if use_mixup and random.random() < 0.25:
            img_emb, txt_emb, lam, index = self.mixup_data(img_emb, txt_emb, 0.15)
        
        # 主要InfoNCE损失
        sim_matrix = torch.mm(img_emb, txt_emb.t()) / self.temperature
        labels = torch.arange(batch_size)
        if img_emb.is_cuda:
            labels = labels.cuda()
        
        loss_i2t = F.cross_entropy(sim_matrix, labels, label_smoothing=self.label_smoothing)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels, label_smoothing=self.label_smoothing)
        loss_infonce = (loss_i2t + loss_t2i) / 2
        
        # 硬负样本三元组损失
        loss_triplet = self.hard_negative_mining(img_emb, txt_emb, epoch, total_epochs)
        
        # 对比学习损失
        loss_contrast = self.contrastive_learning(img_emb, txt_emb)
        
        # 总损失
        total_loss = (self.lambda_infonce * loss_infonce + 
                     self.lambda_triplet * loss_triplet +
                     self.lambda_contrast * loss_contrast)
        
        return total_loss, loss_infonce, loss_triplet, loss_contrast


class AdvancedTrainer:
    """高级训练器"""
    
    def __init__(self):
        self.opt = AdvancedConfig()
        
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
        """训练一个epoch"""
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
            
            img_emb, txt_emb = model.forward_emb(images, captions, lengths)
            total_loss, loss_infonce, loss_triplet, loss_contrast = model.forward_loss(
                img_emb, txt_emb, epoch, self.opt.num_epochs, 
                use_mixup=(epoch > self.opt.curriculum_start)
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)
            optimizer.step()
            
            train_logger.update('Loss', total_loss.item(), img_emb.size(0))
            train_logger.update('InfoNCE', loss_infonce.item(), img_emb.size(0))
            train_logger.update('Triplet', loss_triplet.item(), img_emb.size(0))
            train_logger.update('Contrast', loss_contrast.item(), img_emb.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.opt.log_step == 0:
                print(f'Epoch [{epoch+1}][{i}/{len(data_loader)}] '
                      f'LR: {lr:.6f} '
                      f'Time {batch_time.val:.3f} '
                      f'Loss {train_logger.meters["Loss"].val:.4f} '
                      f'InfoNCE {train_logger.meters["InfoNCE"].val:.4f} '
                      f'Triplet {train_logger.meters["Triplet"].val:.4f} '
                      f'Contrast {train_logger.meters["Contrast"].val:.4f}')
    
    def validate(self, model, data_loader):
        """模型验证"""
        model.eval()
        
        img_embs = []
        txt_embs = []
        
        print("验证中...")
        with torch.no_grad():
            for val_data in data_loader:
                images, captions, lengths, ids = val_data
                
                if torch.cuda.is_available():
                    images = images.cuda()
                    captions = captions.cuda()
                
                img_emb, txt_emb = model.forward_emb(images, captions, lengths)
                
                img_embs.append(img_emb.cpu().numpy())
                txt_embs.append(txt_emb.cpu().numpy())
        
        img_embs = np.concatenate(img_embs, axis=0)
        txt_embs = np.concatenate(txt_embs, axis=0)
        
        (r1, r5, r10, medr, meanr) = i2t(img_embs, txt_embs, measure='cosine')
        (r1i, r5i, r10i, medri, meanri) = t2i(img_embs, txt_embs, measure='cosine')
        
        rsum = r1 + r5 + r10 + r1i + r5i + r10i
        avg_r1 = (r1 + r1i) / 2
        
        print(f"Text to Image: R@1: {r1i:.2f}%, R@5: {r5i:.2f}%, R@10: {r10i:.2f}%")
        print(f"Image to Text: R@1: {r1:.2f}%, R@5: {r5:.2f}%, R@10: {r10:.2f}%")
        print(f"平均 R@1: {avg_r1:.2f}%")
        print(f"R-sum: {rsum:.2f}")
        
        return rsum, avg_r1
    
    def train(self):
        """主训练流程"""
        print("=== 高级优化训练 - 突破37%瓶颈 ===")
        print(f"当前瓶颈: 37%")
        print(f"突破目标: 50%+")
        print(f"核心策略: 多视角+对比学习+硬负样本挖掘")
        
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
        
        model = AdvancedVSEModel(self.opt)
        
        if torch.cuda.is_available():
            model.cuda()
            print("使用GPU")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {total_params:,}")
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.opt.lr_vse,
            weight_decay=self.opt.weight_decay,
            betas=(0.9, 0.999)
        )
        
        best_rsum = 0
        best_avg_r1 = 0
        patience_counter = 0
        
        print(f"\n开始训练 {self.opt.num_epochs} epochs...")
        
        for epoch in range(self.opt.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.opt.num_epochs}")
            print("-" * 60)
            
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
                
                if best_avg_r1 >= 50.0:
                    print(f"🎉 突破50%目标!")
                if best_avg_r1 >= 70.0:
                    print(f"🎉🎉🎉 达到70%最终目标!")
                    break
                    
            else:
                patience_counter += 1
                
            if patience_counter >= self.opt.early_stop_patience:
                print(f"早停")
                break
            
            improvement = best_avg_r1 - 37.47
            remaining_50 = 50.0 - best_avg_r1
            remaining_70 = 70.0 - best_avg_r1
            
            print(f"当前最佳: {best_avg_r1:.2f}%")
            print(f"突破程度: {improvement:.2f}%")
            print(f"距离50%: {remaining_50:.2f}%")
            print(f"距离70%: {remaining_70:.2f}%")
            
            # 动态策略提示
            if epoch > 15 and improvement < 2.0:
                print("⚡ 考虑更激进的优化策略...")
        
        print(f"\n{'='*60}")
        print("高级优化训练完成!")
        print(f"基线: 37.47%")
        print(f"结果: {best_avg_r1:.2f}%")
        print(f"突破: {best_avg_r1 - 37.47:.2f}%")
        
        if best_avg_r1 >= 70.0:
            print("🎉🎉🎉 成功达到70%最终目标!")
        elif best_avg_r1 >= 50.0:
            print("🎉 成功突破50%!")
        elif best_avg_r1 > 42.0:
            print("⚡ 显著提升，继续优化!")
        else:
            print(f"需要更深入优化，距离50%还需: {50.0 - best_avg_r1:.2f}%")


def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = AdvancedTrainer()
    trainer.train()


if __name__ == '__main__':
    main() 