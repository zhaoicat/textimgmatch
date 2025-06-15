#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
渐进式提升训练脚本

基于37%基线的渐进改进策略：
1. 增强模型容量但避免过拟合
2. 改进特征融合和注意力机制  
3. 优化训练策略和正则化
4. 目标：稳步提升到42-45%
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


class BoostConfig:
    """提升配置"""
    def __init__(self):
        # 数据配置
        self.data_path = './data/tk_precomp'
        self.data_name = 'tk_precomp'
        
        # 适度增强的模型配置
        self.embed_size = 1280     # 适度增大
        self.word_dim = 400        # 适度增大
        self.vocab_size = 20000
        self.img_dim = 2048
        
        # 网络配置
        self.num_heads = 10        # 适度增加
        self.num_layers = 3        # 适中深度
        self.dropout = 0.22        # 稍高dropout
        self.hidden_dim = 1600     # 适中隐藏层
        
        # 训练配置
        self.batch_size = 18       # 平衡批次
        self.num_epochs = 150      # 更多轮次
        self.lr_vse = 6e-5         # 适中学习率
        self.workers = 2
        
        # 损失配置
        self.margin = 0.06         # 稍严格边距
        self.temperature = 0.03    # 更低温度
        self.lambda_triplet = 0.7
        self.lambda_infonce = 1.0
        self.lambda_focal = 0.2    # Focal loss权重
        
        # 训练策略
        self.warmup_epochs = 5
        self.cosine_epochs = 40
        self.grad_clip = 0.8
        self.weight_decay = 0.018
        
        # 数据增强
        self.label_smoothing = 0.08
        self.mixup_alpha = 0.1
        
        # 其他配置
        self.log_step = 20
        self.logger_name = './runs/tk_boost/log'
        self.model_name = './runs/tk_boost/checkpoint'
        self.early_stop_patience = 25


class FocalLoss(nn.Module):
    """Focal Loss for hard examples"""
    
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class MultiScaleImageEncoder(nn.Module):
    """多尺度图像编码器"""
    
    def __init__(self, img_dim, embed_size, num_heads=10, dropout=0.22):
        super(MultiScaleImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 多层特征变换
        self.feature_proj = nn.Sequential(
            nn.Linear(img_dim, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size),
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size),
            nn.Dropout(dropout)
        )
        
        # 多尺度注意力池化
        self.global_attention = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, 1)
        )
        
        self.local_attention = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_size // 2, 1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size),
            nn.ReLU(inplace=True),
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
            
            # 前馈网络
            ff_out = self.feed_forward(x)
            x = x + ff_out
            
            # 多尺度池化
            # 全局注意力
            global_weights = self.global_attention(x).squeeze(-1)
            global_weights = F.softmax(global_weights, dim=1)
            global_features = torch.bmm(global_weights.unsqueeze(1), x).squeeze(1)
            
            # 局部注意力（关注重要区域）
            local_weights = self.local_attention(x).squeeze(-1)
            local_weights = F.softmax(local_weights, dim=1)
            local_features = torch.bmm(local_weights.unsqueeze(1), x).squeeze(1)
            
            # 特征融合
            combined = torch.cat([global_features, local_features], dim=-1)
            features = self.fusion(combined)
            
        else:
            features = self.feature_proj(images)
        
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class EnhancedTextEncoder(nn.Module):
    """增强文本编码器"""
    
    def __init__(self, vocab_size, word_dim, embed_size, dropout=0.22):
        super(EnhancedTextEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_dim = word_dim
        
        # 词嵌入
        self.embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)
        self.embed_proj = nn.Linear(word_dim, embed_size)
        self.embed_dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(100, embed_size) * 0.02)
        
        # 多层双向LSTM
        self.lstm1 = nn.LSTM(
            embed_size, embed_size // 2, 1,
            batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            embed_size, embed_size // 2, 1,
            batch_first=True, bidirectional=True
        )
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=10,
            dropout=dropout,
            batch_first=True
        )
        
        # 层级注意力
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
        
        # 词嵌入和投影
        embedded = self.embed(captions)
        embedded = self.embed_proj(embedded)
        
        # 位置编码
        pos_len = min(max_len, self.pos_embedding.size(0))
        embedded[:, :pos_len] += self.pos_embedding[:pos_len].unsqueeze(0)
        embedded = self.embed_dropout(embedded)
        
        # 双层LSTM
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
        
        lstm_out2 = self.layer_norm(lstm_out2 + attn_out)
        
        # 词级注意力
        attention_weights = self.word_attention(lstm_out2).squeeze(-1)
        
        # 长度mask
        mask = torch.zeros(batch_size, max_len)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        
        if lstm_out2.is_cuda:
            mask = mask.cuda()
        
        attention_weights.masked_fill_(mask == 0, -1e9)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 加权求和
        features = torch.bmm(attention_weights.unsqueeze(1), lstm_out2).squeeze(1)
        
        # 特征增强
        features = self.feature_enhance(features)
        features = self.dropout(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


class BoostVSEModel(nn.Module):
    """提升VSE模型"""
    
    def __init__(self, opt):
        super(BoostVSEModel, self).__init__()
        
        self.img_enc = MultiScaleImageEncoder(
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
        
        # 损失函数
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
        self.margin = opt.margin
        self.temperature = opt.temperature
        self.lambda_triplet = opt.lambda_triplet
        self.lambda_infonce = opt.lambda_infonce
        self.lambda_focal = opt.lambda_focal
        self.label_smoothing = opt.label_smoothing
    
    def forward_emb(self, images, captions, lengths):
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb
    
    def mixup_data(self, img_emb, cap_emb, alpha=0.1):
        """Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = img_emb.size(0)
        index = torch.randperm(batch_size)
        if img_emb.is_cuda:
            index = index.cuda()
        
        mixed_img = lam * img_emb + (1 - lam) * img_emb[index]
        mixed_cap = lam * cap_emb + (1 - lam) * cap_emb[index]
        
        return mixed_img, mixed_cap, lam, index
    
    def forward_loss(self, img_emb, cap_emb, use_mixup=False):
        """计算增强损失"""
        batch_size = img_emb.size(0)
        
        # Mixup增强
        if use_mixup and random.random() < 0.3:
            img_emb, cap_emb, lam, index = self.mixup_data(img_emb, cap_emb, 0.1)
        
        # InfoNCE损失 + 标签平滑
        sim_matrix = torch.mm(img_emb, cap_emb.t()) / self.temperature
        labels = torch.arange(batch_size)
        if img_emb.is_cuda:
            labels = labels.cuda()
        
        loss_i2t = F.cross_entropy(sim_matrix, labels, label_smoothing=self.label_smoothing)
        loss_t2i = F.cross_entropy(sim_matrix.t(), labels, label_smoothing=self.label_smoothing)
        loss_infonce = (loss_i2t + loss_t2i) / 2
        
        # Focal Loss for hard examples
        loss_focal_i2t = self.focal_loss(sim_matrix, labels)
        loss_focal_t2i = self.focal_loss(sim_matrix.t(), labels)
        loss_focal = (loss_focal_i2t + loss_focal_t2i) / 2
        
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
        
        # 总损失
        total_loss = (self.lambda_infonce * loss_infonce + 
                     self.lambda_triplet * loss_triplet +
                     self.lambda_focal * loss_focal)
        
        return total_loss, loss_infonce, loss_triplet, loss_focal


class BoostTrainer:
    """提升训练器"""
    
    def __init__(self):
        self.opt = BoostConfig()
        
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
                lr = self.opt.lr_vse * 0.15 * (0.95 ** decay_epochs)
        
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
            total_loss, loss_infonce, loss_triplet, loss_focal = model.forward_loss(
                img_emb, cap_emb, use_mixup=(epoch > 10)
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt.grad_clip)
            optimizer.step()
            
            train_logger.update('Loss', total_loss.item(), img_emb.size(0))
            train_logger.update('InfoNCE', loss_infonce.item(), img_emb.size(0))
            train_logger.update('Triplet', loss_triplet.item(), img_emb.size(0))
            train_logger.update('Focal', loss_focal.item(), img_emb.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.opt.log_step == 0:
                print(f'Epoch [{epoch+1}][{i}/{len(data_loader)}] '
                      f'LR: {lr:.6f} '
                      f'Time {batch_time.val:.3f} '
                      f'Loss {train_logger.meters["Loss"].val:.4f} '
                      f'InfoNCE {train_logger.meters["InfoNCE"].val:.4f} '
                      f'Triplet {train_logger.meters["Triplet"].val:.4f} '
                      f'Focal {train_logger.meters["Focal"].val:.4f}')
    
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
        print("=== 渐进式提升训练 ===")
        print(f"基线: 37.47%")
        print(f"目标: 42-45%")
        print(f"策略: 多尺度特征+增强损失+数据增强")
        
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
        
        model = BoostVSEModel(self.opt)
        
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
            remaining = 45.0 - best_avg_r1
            
            print(f"当前最佳: {best_avg_r1:.2f}%")
            print(f"改进: {improvement:.2f}%")
            print(f"距离45%: {remaining:.2f}%")
        
        print(f"\n{'='*50}")
        print("提升训练完成!")
        print(f"基线: 37.47%")
        print(f"结果: {best_avg_r1:.2f}%")
        print(f"提升: {best_avg_r1 - 37.47:.2f}%")
        
        if best_avg_r1 >= 45.0:
            print("🎉 成功达到45%目标!")
        elif best_avg_r1 > 40.0:
            print("⚡ 显著提升，继续努力!")
        else:
            print(f"距离45%还需: {45.0 - best_avg_r1:.2f}%")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = BoostTrainer()
    trainer.train()


if __name__ == '__main__':
    main() 