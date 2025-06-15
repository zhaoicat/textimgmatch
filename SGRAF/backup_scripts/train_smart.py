#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能优化训练脚本 - 针对37%瓶颈的精准突破

问题分析：
1. 37%瓶颈 = 架构限制 + 数据不足 + 优化不当
2. 需要：更好的特征表示 + 更强的匹配机制 + 更优的训练策略

解决方案：
1. 双路径特征提取（粗粒度+细粒度）
2. 自适应相似度学习
3. 课程学习 + 困难样本重点训练
4. 集成多种损失函数的优势

目标：稳定突破到45-50%
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


class SmartConfig:
    """智能配置 - 基于问题分析优化"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 平衡的模型配置
        self.embed_size = 768       # 经典配置
        self.word_dim = 300         # 标准词嵌入
        self.vocab_size = 20000
        self.img_dim = 2048
        
        # 网络配置
        self.num_heads = 8
        self.dropout = 0.3          # 适度正则化
        self.hidden_dim = 1024
        
        # 训练配置
        self.batch_size = 24
        self.num_epochs = 100
        self.lr_vse = 8e-5          # 稍大学习率
        self.workers = 2
        
        # 损失配置
        self.margin = 0.1           # 适中边距
        self.temperature = 0.07     # 适中温度
        self.alpha_triplet = 0.5    # 三元组权重
        self.alpha_infonce = 1.0    # InfoNCE权重
        self.alpha_smooth = 0.3     # 平滑权重
        
        # 智能训练策略
        self.warmup_epochs = 6
        self.curriculum_start = 8   # 课程学习开始
        self.hard_mining_start = 15 # 困难挖掘开始
        
        # 正则化
        self.grad_clip = 1.0
        self.weight_decay = 0.015
        self.label_smoothing = 0.05
        
        # 其他
        self.log_step = 20
        self.logger_name = './runs/tk_smart/log'
        self.model_name = './runs/tk_smart/checkpoint'
        self.early_stop_patience = 20


class DualPathImageEncoder(nn.Module):
    """双路径图像编码器 - 粗粒度+细粒度"""
    
    def __init__(self, img_dim, embed_size, dropout=0.3):
        super(DualPathImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 粗粒度路径（全局特征）
        self.coarse_path = nn.Sequential(
            nn.Linear(img_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size),
            nn.LayerNorm(embed_size)
        )
        
        # 细粒度路径（区域特征）
        self.fine_path = nn.Sequential(
            nn.Linear(img_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(36, embed_size) * 0.02)
        
        # 区域注意力
        self.region_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 自适应融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.Sigmoid()
        )
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_size * 3, embed_size),  # 修改为3倍维度
            nn.LayerNorm(embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size)
        )
        
        # 注意力池化
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.Tanh(),
            nn.Linear(embed_size // 2, 1)
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
            # 粗粒度特征（全局）
            global_features = images.mean(dim=1)  # (batch, 2048)
            coarse_emb = self.coarse_path(global_features)  # (batch, embed_size)
            
            # 细粒度特征（区域）
            fine_features = self.fine_path(images)  # (batch, 36, embed_size)
            
            # 位置编码
            fine_features = fine_features + self.pos_embedding.unsqueeze(0)
            fine_features = self.dropout(fine_features)
            
            # 区域注意力
            attn_features, _ = self.region_attention(
                fine_features, fine_features, fine_features
            )
            fine_features = fine_features + attn_features
            
            # 注意力池化
            attention_weights = self.attention_pool(fine_features).squeeze(-1)
            attention_weights = F.softmax(attention_weights, dim=1)
            fine_emb = torch.bmm(attention_weights.unsqueeze(1), fine_features).squeeze(1)
            
            # 自适应融合
            combined = torch.cat([coarse_emb, fine_emb], dim=-1)
            gate = self.fusion_gate(combined)
            fused_features = gate * coarse_emb + (1 - gate) * fine_emb
            
            # 最终投影
            final_combined = torch.cat([fused_features, coarse_emb, fine_emb], dim=-1)
            features = self.fusion_proj(final_combined)
            
        else:
            features = self.coarse_path(images)
        
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class SmartTextEncoder(nn.Module):
    """智能文本编码器 - 层次化语义理解"""
    
    def __init__(self, vocab_size, word_dim, embed_size, dropout=0.3):
        super(SmartTextEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 词嵌入
        self.embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 词级别编码
        self.word_encoder = nn.Sequential(
            nn.Linear(word_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(50, embed_size) * 0.02)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            embed_size, embed_size // 2, 2,
            batch_first=True, bidirectional=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 层级注意力
        self.hierarchical_attention = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, 1)
        )
        
        # 语义增强
        self.semantic_enhance = nn.Sequential(
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
        
        # 词嵌入
        embedded = self.embed(captions)
        embedded = self.embed_dropout(embedded)
        
        # 词级编码
        word_features = self.word_encoder(embedded)
        
        # 位置编码
        pos_len = min(max_len, self.pos_embedding.size(0))
        word_features[:, :pos_len] += self.pos_embedding[:pos_len].unsqueeze(0)
        
        # LSTM编码
        packed = nn.utils.rnn.pack_padded_sequence(
            word_features, lengths, batch_first=True, enforce_sorted=False
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
        
        # 层级注意力
        attention_weights = self.hierarchical_attention(lstm_out).squeeze(-1)
        
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
        
        # 语义增强
        features = self.semantic_enhance(features)
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class AdaptiveSimilarityLearning(nn.Module):
    """自适应相似度学习"""
    
    def __init__(self, embed_size, temperature=0.07):
        super(AdaptiveSimilarityLearning, self).__init__()
        self.temperature = temperature
        
        # 自适应相似度计算
        self.similarity_net = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_size, 1),
            nn.Sigmoid()
        )
        
        # 温度学习
        self.temp_net = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img_emb, txt_emb):
        batch_size = img_emb.size(0)
        
        # 计算所有配对的特征
        img_expanded = img_emb.unsqueeze(1).expand(-1, batch_size, -1)  # (B, B, D)
        txt_expanded = txt_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, B, D)
        
        paired_features = torch.cat([img_expanded, txt_expanded], dim=-1)  # (B, B, 2D)
        
        # 自适应相似度
        adaptive_sim = self.similarity_net(paired_features).squeeze(-1)  # (B, B)
        
        # 自适应温度
        adaptive_temp = self.temp_net(paired_features).squeeze(-1)  # (B, B)
        adaptive_temp = 0.01 + 0.2 * adaptive_temp  # 限制温度范围
        
        # 基础余弦相似度
        cosine_sim = torch.mm(img_emb, txt_emb.t())
        
        # 融合相似度
        final_sim = 0.7 * cosine_sim + 0.3 * adaptive_sim
        final_sim = final_sim / adaptive_temp
        
        return final_sim


class SmartVSEModel(nn.Module):
    """智能VSE模型"""
    
    def __init__(self, opt):
        super(SmartVSEModel, self).__init__()
        
        # 编码器
        self.img_enc = DualPathImageEncoder(
            img_dim=opt.img_dim,
            embed_size=opt.embed_size,
            dropout=opt.dropout
        )
        
        self.txt_enc = SmartTextEncoder(
            vocab_size=opt.vocab_size,
            word_dim=opt.word_dim,
            embed_size=opt.embed_size,
            dropout=opt.dropout
        )
        
        # 自适应相似度学习
        self.adaptive_sim = AdaptiveSimilarityLearning(
            embed_size=opt.embed_size,
            temperature=opt.temperature
        )
        
        # 损失参数
        self.margin = opt.margin
        self.temperature = opt.temperature
        self.alpha_triplet = opt.alpha_triplet
        self.alpha_infonce = opt.alpha_infonce
        self.alpha_smooth = opt.alpha_smooth
        self.label_smoothing = opt.label_smoothing
    
    def forward_emb(self, images, captions, lengths):
        img_emb = self.img_enc(images)
        txt_emb = self.txt_enc(captions, lengths)
        return img_emb, txt_emb
    
    def forward_loss(self, img_emb, txt_emb, epoch=0, total_epochs=100):
        """智能损失计算"""
        batch_size = img_emb.size(0)
        
        # 自适应相似度
        adaptive_scores = self.adaptive_sim(img_emb, txt_emb)
        
        # InfoNCE损失（使用自适应相似度）
        labels = torch.arange(batch_size)
        if img_emb.is_cuda:
            labels = labels.cuda()
        
        loss_i2t = F.cross_entropy(adaptive_scores, labels, label_smoothing=self.label_smoothing)
        loss_t2i = F.cross_entropy(adaptive_scores.t(), labels, label_smoothing=self.label_smoothing)
        loss_infonce = (loss_i2t + loss_t2i) / 2
        
        # 传统三元组损失
        cosine_scores = torch.mm(img_emb, txt_emb.t())
        diagonal = cosine_scores.diag().view(-1, 1)
        
        d1 = diagonal.expand_as(cosine_scores)
        cost_s = (self.margin + cosine_scores - d1).clamp(min=0)
        
        d2 = diagonal.t().expand_as(cosine_scores)
        cost_im = (self.margin + cosine_scores - d2).clamp(min=0)
        
        mask = torch.eye(batch_size) > 0.5
        if cosine_scores.is_cuda:
            mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        
        # 课程学习：逐渐关注困难样本
        if epoch >= 15:  # 困难挖掘阶段
            progress = min(1.0, (epoch - 15) / 30)
            top_k = max(1, int(batch_size * (0.3 + 0.5 * progress)))
            cost_s_top, _ = cost_s.topk(top_k, dim=1)
            cost_im_top, _ = cost_im.topk(top_k, dim=0)
            loss_triplet = cost_s_top.mean() + cost_im_top.mean()
        else:
            loss_triplet = cost_s.mean() + cost_im.mean()
        
        # 平滑损失（一致性正则化）
        img_norm = F.normalize(img_emb, p=2, dim=1)
        txt_norm = F.normalize(txt_emb, p=2, dim=1)
        consistency_loss = F.mse_loss(img_norm, txt_norm)
        
        # 总损失
        total_loss = (self.alpha_infonce * loss_infonce + 
                     self.alpha_triplet * loss_triplet +
                     self.alpha_smooth * consistency_loss)
        
        return total_loss, loss_infonce, loss_triplet, consistency_loss


class SmartTrainer:
    """智能训练器"""
    
    def __init__(self):
        self.opt = SmartConfig()
        
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
        """智能学习率调度"""
        if epoch < self.opt.warmup_epochs:
            lr = self.opt.lr_vse * (epoch + 1) / self.opt.warmup_epochs
        else:
            # 分阶段衰减
            if epoch < 30:
                lr = self.opt.lr_vse
            elif epoch < 60:
                lr = self.opt.lr_vse * 0.5
            else:
                lr = self.opt.lr_vse * 0.2
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def train_epoch(self, model, data_loader, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        
        batch_time = AverageMeter()
        train_logger = LogCollector()
        
        end = time.time()
        
        for i, train_data in enumerate(data_loader):
            images, captions, lengths, ids = train_data
            
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            
            lr = self.adjust_learning_rate(optimizer, epoch)
            
            img_emb, txt_emb = model.forward_emb(images, captions, lengths)
            total_loss, loss_infonce, loss_triplet, loss_smooth = model.forward_loss(
                img_emb, txt_emb, epoch, self.opt.num_epochs
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)
            optimizer.step()
            
            train_logger.update('Loss', total_loss.item(), img_emb.size(0))
            train_logger.update('InfoNCE', loss_infonce.item(), img_emb.size(0))
            train_logger.update('Triplet', loss_triplet.item(), img_emb.size(0))
            train_logger.update('Smooth', loss_smooth.item(), img_emb.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.opt.log_step == 0:
                print(f'Epoch [{epoch+1}][{i}/{len(data_loader)}] '
                      f'LR: {lr:.6f} '
                      f'Time {batch_time.val:.3f} '
                      f'Loss {train_logger.meters["Loss"].val:.4f} '
                      f'InfoNCE {train_logger.meters["InfoNCE"].val:.4f} '
                      f'Triplet {train_logger.meters["Triplet"].val:.4f} '
                      f'Smooth {train_logger.meters["Smooth"].val:.4f}')
    
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
        print("=== 智能优化训练 - 精准突破37%瓶颈 ===")
        print(f"基线: 37.47%")
        print(f"目标: 45-50%")
        print(f"策略: 双路径特征+自适应相似度+课程学习")
        
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
        
        model = SmartVSEModel(self.opt)
        
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
            
            # 训练阶段提示
            if epoch == self.opt.curriculum_start:
                print("🎓 开始课程学习阶段")
            elif epoch == self.opt.hard_mining_start:
                print("⚡ 开始困难样本挖掘")
            
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
                
                # 目标检查
                if best_avg_r1 >= 50.0:
                    print(f"🎉 突破50%!")
                elif best_avg_r1 >= 45.0:
                    print(f"🎉 达到45%目标!")
                elif best_avg_r1 >= 70.0:
                    print(f"🎉🎉🎉 达到70%最终目标!")
                    break
                    
            else:
                patience_counter += 1
                
            if patience_counter >= self.opt.early_stop_patience:
                print(f"早停")
                break
            
            improvement = best_avg_r1 - 37.47
            remaining = 50.0 - best_avg_r1
            
            print(f"当前最佳: {best_avg_r1:.2f}%")
            print(f"突破: {improvement:.2f}%")
            print(f"距离50%: {remaining:.2f}%")
            
            # 智能建议
            if epoch > 10 and improvement < 1.0:
                print("💡 建议: 考虑调整学习率或模型架构")
            elif improvement > 5.0:
                print("🚀 表现优秀，继续保持!")
        
        print(f"\n{'='*50}")
        print("智能优化训练完成!")
        print(f"基线: 37.47%")
        print(f"结果: {best_avg_r1:.2f}%")
        print(f"突破: {best_avg_r1 - 37.47:.2f}%")
        
        if best_avg_r1 >= 70.0:
            print("🎉🎉🎉 达到70%最终目标!")
        elif best_avg_r1 >= 50.0:
            print("🎉 成功突破50%!")
        elif best_avg_r1 >= 45.0:
            print("🎉 达到45%目标!")
        elif best_avg_r1 > 40.0:
            print("⚡ 显著提升!")
        else:
            print(f"需要进一步优化，距离45%: {45.0 - best_avg_r1:.2f}%")


def main():
    """主函数"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = SmartTrainer()
    trainer.train()


if __name__ == '__main__':
    main() 