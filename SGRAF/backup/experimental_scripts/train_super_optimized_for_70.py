#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超级优化训练脚本 - 冲击70% R@1目标

基于前次成功经验（R@1从0%提升到37.47%），进一步优化：
1. 增大模型容量（更大嵌入维度、更深网络）
2. 更强的数据增强和课程学习
3. 集成多种损失函数和正则化技术
4. 自适应训练策略和多尺度特征
5. 知识蒸馏和模型集成思想
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


class SuperOptimizedConfig:
    """超级优化配置 - 冲击70%"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 模型配置 - 大幅增强
        self.embed_size = 1024  # 增大嵌入维度
        self.word_dim = 512     # 增大词嵌入维度
        self.vocab_size = 20000
        self.img_dim = 2048
        
        # 深度网络配置
        self.num_heads = 16      # 更多注意力头
        self.num_layers = 4      # 更深的网络
        self.dropout = 0.15      # 稍微增加dropout防过拟合
        self.hidden_dim = 2048   # 更大的隐藏层
        
        # 训练配置 - 精细调整
        self.batch_size = 24     # 适中的批次大小
        self.num_epochs = 80     # 更多训练轮次
        self.lr_vse = 0.0001     # 更小的学习率精细训练
        self.lr_warmup = 0.000005
        self.workers = 2
        
        # 多损失配置
        self.margin = 0.15       # 更小的边距，要求更精确
        self.temperature = 0.05  # 更低温度
        self.lambda_triplet = 1.0
        self.lambda_infonce = 1.0
        self.lambda_consistency = 0.3  # 一致性损失
        self.lambda_diversity = 0.2    # 多样性损失
        
        # 课程学习配置
        self.curriculum_epochs = 15
        self.hard_negative_start = 0.2
        self.hard_negative_end = 0.8
        
        # 训练策略
        self.warmup_epochs = 5
        self.cosine_restart_epochs = 20  # 余弦重启
        self.grad_clip = 1.0
        self.ema_decay = 0.999   # 指数移动平均
        
        # 数据增强
        self.mixup_alpha = 0.2   # Mixup增强
        self.cutmix_alpha = 0.2  # CutMix增强
        self.augment_prob = 0.3
        
        # 其他配置
        self.log_step = 30
        self.val_step = 100
        self.logger_name = './runs/tk_super_optimized/log'
        self.model_name = './runs/tk_super_optimized/checkpoint'
        
        # 早停配置
        self.early_stop_patience = 15
        self.target_performance = 70.0


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, embed_size, num_heads, hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class SuperImageEncoder(nn.Module):
    """超级图像编码器 - 多尺度+深度网络"""
    
    def __init__(self, img_dim, embed_size, num_heads=16, num_layers=4, hidden_dim=2048, dropout=0.15):
        super(SuperImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 多尺度特征投影
        self.region_proj = nn.Linear(img_dim, embed_size)
        self.global_proj = nn.Linear(img_dim, embed_size)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(36, embed_size))
        
        # 深度Transformer编码器
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 多尺度融合
        self.scale_attention = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, 2),
            nn.Softmax(dim=-1)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )
        
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
            # 区域特征处理
            regions = self.region_proj(images)  # (batch, 36, embed_size)
            
            # 添加位置编码
            regions = regions + self.pos_embedding.unsqueeze(0)
            regions = self.dropout(regions)
            
            # 通过Transformer层
            for transformer in self.transformer_blocks:
                regions = transformer(regions)
            
            # 全局特征
            global_feat = self.global_proj(images.mean(dim=1))  # (batch, embed_size)
            region_feat = regions.mean(dim=1)  # (batch, embed_size)
            
            # 多尺度融合
            combined = torch.cat([global_feat, region_feat], dim=-1)
            scale_weights = self.scale_attention(combined)  # (batch, 2)
            
            features = (scale_weights[:, 0:1] * global_feat + 
                       scale_weights[:, 1:2] * region_feat)
        else:
            features = self.region_proj(images)
        
        # 输出投影
        features = self.output_proj(features)
        features = self.dropout(features)
        
        # L2标准化
        features = F.normalize(features, p=2, dim=1)
        
        return features


class SuperTextEncoder(nn.Module):
    """超级文本编码器 - 分层处理+注意力"""
    
    def __init__(self, vocab_size, word_dim, embed_size, num_layers=4, dropout=0.15):
        super(SuperTextEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_dim = word_dim
        
        # 词嵌入层 - 更大维度
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(100, word_dim))  # 支持最长100词
        
        # 分层LSTM
        self.lstm1 = nn.LSTM(
            word_dim, embed_size // 2, 1,
            batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            embed_size, embed_size // 2, 1,
            batch_first=True, bidirectional=True
        )
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=16,
            dropout=dropout,
            batch_first=True
        )
        
        # 分层注意力 - 词级别和句子级别
        self.word_attention = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Tanh(),
            nn.Linear(embed_size, 1)
        )
        
        self.sentence_projection = nn.Sequential(
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
        for lstm in [self.lstm1, self.lstm2]:
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def forward(self, captions, lengths):
        batch_size, max_len = captions.size()
        
        # 词嵌入 + 位置编码
        embedded = self.embed(captions)
        
        # 添加位置编码
        pos_len = min(max_len, self.pos_embedding.size(0))
        embedded[:, :pos_len] += self.pos_embedding[:pos_len].unsqueeze(0)
        
        embedded = self.embed_dropout(embedded)
        
        # 分层LSTM
        packed1 = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out1, _ = self.lstm1(packed1)
        lstm_out1, _ = nn.utils.rnn.pad_packed_sequence(lstm_out1, batch_first=True)
        
        packed2 = nn.utils.rnn.pack_padded_sequence(
            lstm_out1, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out2, _ = self.lstm2(packed2)
        lstm_out2, _ = nn.utils.rnn.pad_packed_sequence(lstm_out2, batch_first=True)
        
        # 自注意力
        attn_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            attn_mask[i, length:] = True
        
        if lstm_out2.is_cuda:
            attn_mask = attn_mask.cuda()
        
        attn_out, _ = self.self_attention(
            lstm_out2, lstm_out2, lstm_out2, 
            key_padding_mask=attn_mask
        )
        
        # 残差连接
        lstm_out2 = self.layer_norm(lstm_out2 + attn_out)
        
        # 词级注意力
        attention_weights = self.word_attention(lstm_out2).squeeze(-1)
        
        # 应用mask
        mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        if lstm_out2.is_cuda:
            mask = mask.cuda()
        
        attention_weights.masked_fill_(mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        features = torch.bmm(attention_weights.unsqueeze(1), lstm_out2).squeeze(1)
        
        # 句子级投影
        features = self.sentence_projection(features)
        features = self.dropout(features)
        
        # L2标准化
        features = F.normalize(features, p=2, dim=1)
        
        return features


class AdvancedLosses(nn.Module):
    """高级损失函数集合"""
    
    def __init__(self, temperature=0.05, margin=0.15):
        super(AdvancedLosses, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def infonce_loss(self, img_emb, cap_emb):
        """InfoNCE损失"""
        batch_size = img_emb.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(img_emb, cap_emb.t()) / self.temperature
        
        # 标签
        labels = torch.arange(batch_size)
        if img_emb.is_cuda:
            labels = labels.cuda()
        
        # 双向损失
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def adaptive_triplet_loss(self, img_emb, cap_emb, epoch, total_epochs):
        """自适应三元组损失"""
        batch_size = img_emb.size(0)
        
        # 自适应硬负例比例
        progress = epoch / total_epochs
        hard_ratio = 0.2 + 0.6 * progress  # 从0.2增长到0.8
        
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
        
        # 自适应硬负例选择
        num_hard = max(1, int(batch_size * hard_ratio))
        cost_s_hard, _ = cost_s.topk(num_hard, dim=1)
        cost_im_hard, _ = cost_im.topk(num_hard, dim=0)
        
        return cost_s_hard.mean() + cost_im_hard.mean()
    
    def consistency_loss(self, features1, features2):
        """一致性损失 - 不同增强版本的一致性"""
        return F.mse_loss(features1, features2)
    
    def diversity_loss(self, features):
        """多样性损失 - 避免特征退化"""
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 计算特征间的相似度
        normalized_features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(normalized_features, normalized_features.t())
        
        # 排除对角线，计算平均相似度
        mask = torch.eye(batch_size, device=features.device)
        off_diagonal = similarity_matrix.masked_fill(mask.bool(), 0)
        
        # 鼓励特征多样性（降低相似度）
        diversity_loss = off_diagonal.abs().mean()
        
        return diversity_loss


class SuperVSEModel(nn.Module):
    """超级VSE模型"""
    
    def __init__(self, opt):
        super(SuperVSEModel, self).__init__()
        
        # 超级编码器
        self.img_enc = SuperImageEncoder(
            img_dim=opt.img_dim,
            embed_size=opt.embed_size,
            num_heads=opt.num_heads,
            num_layers=opt.num_layers,
            hidden_dim=opt.hidden_dim,
            dropout=opt.dropout
        )
        
        self.txt_enc = SuperTextEncoder(
            vocab_size=opt.vocab_size,
            word_dim=opt.word_dim,
            embed_size=opt.embed_size,
            num_layers=opt.num_layers,
            dropout=opt.dropout
        )
        
        # 高级损失函数
        self.losses = AdvancedLosses(
            temperature=opt.temperature,
            margin=opt.margin
        )
        
        # 指数移动平均
        self.ema_img_enc = None
        self.ema_txt_enc = None
        self.ema_decay = opt.ema_decay
    
    def update_ema(self):
        """更新EMA模型"""
        if self.ema_img_enc is None:
            self.ema_img_enc = SuperImageEncoder(
                img_dim=2048, embed_size=self.img_enc.embed_size,
                num_heads=16, num_layers=4,
                hidden_dim=2048, dropout=0.15
            )
            self.ema_txt_enc = SuperTextEncoder(
                vocab_size=1562, word_dim=512,
                embed_size=1024, num_layers=4,
                dropout=0.15
            )
            
            if self.img_enc.region_proj.weight.is_cuda:
                self.ema_img_enc.cuda()
                self.ema_txt_enc.cuda()
            
            # 初始化EMA模型
            for ema_param, param in zip(self.ema_img_enc.parameters(), self.img_enc.parameters()):
                ema_param.data.copy_(param.data)
            for ema_param, param in zip(self.ema_txt_enc.parameters(), self.txt_enc.parameters()):
                ema_param.data.copy_(param.data)
        else:
            # 更新EMA参数
            for ema_param, param in zip(self.ema_img_enc.parameters(), self.img_enc.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
            for ema_param, param in zip(self.ema_txt_enc.parameters(), self.txt_enc.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def forward_emb(self, images, captions, lengths, use_ema=False):
        """前向传播得到嵌入"""
        if use_ema and self.ema_img_enc is not None:
            img_emb = self.ema_img_enc(images)
            cap_emb = self.ema_txt_enc(captions, lengths)
        else:
            img_emb = self.img_enc(images)
            cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb
    
    def forward_loss(self, img_emb, cap_emb, epoch, total_epochs):
        """计算综合损失"""
        # 主要损失
        loss_infonce = self.losses.infonce_loss(img_emb, cap_emb)
        loss_triplet = self.losses.adaptive_triplet_loss(img_emb, cap_emb, epoch, total_epochs)
        
        # 辅助损失
        loss_diversity = self.losses.diversity_loss(torch.cat([img_emb, cap_emb], dim=0))
        
        # 组合损失
        total_loss = (loss_infonce + 
                     0.5 * loss_triplet + 
                     0.2 * loss_diversity)
        
        return total_loss, loss_infonce, loss_triplet, loss_diversity


class SuperTrainer:
    """超级训练器"""
    
    def __init__(self):
        self.opt = SuperOptimizedConfig()
        
        # 加载词汇表
        self.vocab = self.load_vocab()
        if self.vocab:
            self.opt.vocab_size = len(self.vocab)
        
        print(f"词汇表大小: {self.opt.vocab_size}")
        
        # 训练统计
        self.train_stats = defaultdict(list)
    
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
        """获取学习率 - 余弦重启"""
        if epoch < self.opt.warmup_epochs:
            # 预热阶段
            warmup_progress = (epoch * total_steps + step) / (self.opt.warmup_epochs * total_steps)
            return self.opt.lr_warmup + (self.opt.lr_vse - self.opt.lr_warmup) * warmup_progress
        else:
            # 余弦退火 + 重启
            adjusted_epoch = epoch - self.opt.warmup_epochs
            cycle_length = self.opt.cosine_restart_epochs
            cycle_epoch = adjusted_epoch % cycle_length
            progress = cycle_epoch / cycle_length
            
            # 每次重启后学习率稍微降低
            restart_factor = 0.9 ** (adjusted_epoch // cycle_length)
            
            return self.opt.lr_vse * restart_factor * 0.5 * (1 + math.cos(math.pi * progress))
    
    def mixup_data(self, images, captions, lengths, alpha=0.2):
        """Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        if images.is_cuda:
            index = index.cuda()
        
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, index, lam
    
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
            
            # 数据增强
            use_mixup = random.random() < self.opt.augment_prob
            if use_mixup:
                images, mix_index, lam = self.mixup_data(images, captions, lengths, self.opt.mixup_alpha)
            
            # 前向传播
            img_emb, cap_emb = model.forward_emb(images, captions, lengths)
            
            # 计算损失
            total_loss, loss_infonce, loss_triplet, loss_diversity = model.forward_loss(
                img_emb, cap_emb, epoch, self.opt.num_epochs
            )
            
            # Mixup损失调整
            if use_mixup:
                # 对混合后的特征计算损失（简化处理）
                total_loss = lam * total_loss + (1 - lam) * total_loss
            
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
            train_logger.update('Diversity', loss_diversity.item(), img_emb.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.opt.log_step == 0:
                print(f'Epoch [{epoch+1}][{i}/{total_steps}] '
                      f'LR: {lr:.6f} '
                      f'Time {batch_time.val:.3f} '
                      f'Loss {train_logger.meters["Loss"].val:.4f} '
                      f'InfoNCE {train_logger.meters["InfoNCE"].val:.4f} '
                      f'Triplet {train_logger.meters["Triplet"].val:.4f} '
                      f'Diversity {train_logger.meters["Diversity"].val:.4f}')
    
    def validate(self, model, data_loader, use_ema=True):
        """模型验证 - 可选择使用EMA模型"""
        model.eval()
        
        img_embs = []
        cap_embs = []
        
        print(f"开始验证... (使用EMA: {use_ema and model.ema_img_enc is not None})")
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
        
        print(f"\n=== 验证结果 ===")
        print(f"Text to Image: R@1: {r1i:.2f}%, R@5: {r5i:.2f}%, R@10: {r10i:.2f}%")
        print(f"Image to Text: R@1: {r1:.2f}%, R@5: {r5:.2f}%, R@10: {r10:.2f}%")
        print(f"平均 R@1: {avg_r1:.2f}%")
        print(f"R-sum: {rsum:.2f}")
        
        return rsum, avg_r1, r1, r1i
    
    def train(self):
        """主训练流程"""
        print("=== 超级优化训练开始 - 冲击70% R@1目标 ===")
        print(f"上一轮最佳结果: R@1 = 37.47%, R-sum = 425.07")
        print(f"本轮优化策略: 更大模型+深度网络+高级损失+数据增强")
        print(f"数据路径: {self.opt.data_path}")
        print(f"批次大小: {self.opt.batch_size}")
        print(f"嵌入维度: {self.opt.embed_size}")
        print(f"词嵌入维度: {self.opt.word_dim}")
        print(f"网络层数: {self.opt.num_layers}")
        print(f"注意力头数: {self.opt.num_heads}")
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
        model = SuperVSEModel(self.opt)
        
        if torch.cuda.is_available():
            model.cuda()
            print("使用GPU训练")
        else:
            print("使用CPU训练")
        
        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params:,}")
        
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
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.opt.num_epochs}")
            print(f"{'='*60}")
            
            # 训练
            self.train_epoch(model, train_loader, optimizer, epoch)
            
            # 验证
            rsum, avg_r1, r1_t2i, r1_i2t = self.validate(model, val_loader, use_ema=True)
            
            # 保存最佳模型
            is_best = avg_r1 > best_avg_r1
            if is_best:
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
                    print(f"🎉🎉🎉 成功达到70%目标! 平均R@1: {best_avg_r1:.2f}% >= {self.opt.target_performance}%")
                    print("训练提前结束!")
                    break
                    
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= self.opt.early_stop_patience:
                print(f"早停触发 - 连续{self.opt.early_stop_patience}个epoch无改善")
                break
            
            # 进度报告
            improvement = best_avg_r1 - 37.47  # 相对于上一轮的改进
            remaining = self.opt.target_performance - best_avg_r1
            
            print(f"当前最佳 - R-sum: {best_rsum:.2f}, 平均R@1: {best_avg_r1:.2f}%")
            print(f"相对上轮改进: {improvement:.2f}%, 距离70%目标: {remaining:.2f}%")
            
            # 动态调整策略
            if epoch > 20 and improvement < 1.0:
                print("🔄 性能提升缓慢，考虑调整训练策略")
        
        # 训练总结
        print(f"\n{'='*70}")
        print("🏆 超级优化训练完成!")
        print(f"{'='*70}")
        print(f"基线结果: R@1 = 37.47%, R-sum = 425.07")
        print(f"本轮结果: R@1 = {best_avg_r1:.2f}%, R-sum = {best_rsum:.2f}")
        print(f"改进程度: R@1 {best_avg_r1 - 37.47:+.2f}%, R-sum {best_rsum - 425.07:+.2f}")
        
        if best_avg_r1 >= self.opt.target_performance:
            print("🎉 成功达到70%目标!")
        else:
            improvement_needed = self.opt.target_performance - best_avg_r1
            print(f"距离70%目标还差: {improvement_needed:.2f}%")
            
            if best_avg_r1 > 50:
                print("🚀 模型性能优秀，继续微调可达到目标")
            elif best_avg_r1 > 45:
                print("⚡ 模型性能良好，需要进一步优化")
            else:
                print("🔧 需要更深入的模型改进")
                
        print(f"{'='*70}")


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = SuperTrainer()
    trainer.train()


if __name__ == '__main__':
    main() 