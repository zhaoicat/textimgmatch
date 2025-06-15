"""VSE模型定义"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class ImageEncoder(nn.Module):
    """图像编码器"""
    
    def __init__(self, embed_size, cnn_type='resnet152', finetune=False):
        super(ImageEncoder, self).__init__()
        self.embed_size = embed_size
        
        # 加载预训练CNN模型
        if cnn_type == 'resnet152':
            resnet = models.resnet152(pretrained=True)
            modules = list(resnet.children())[:-1]  # 移除最后的分类层
            self.cnn = nn.Sequential(*modules)
            cnn_dim = 2048
        elif cnn_type == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            cnn_dim = 2048
        else:
            raise ValueError(f"Unsupported CNN type: {cnn_type}")
        
        # 设置CNN参数是否可训练
        for param in self.cnn.parameters():
            param.requires_grad = finetune
        
        # 全连接层映射到嵌入空间
        self.fc = nn.Linear(cnn_dim, embed_size)
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, images):
        """前向传播"""
        # 提取CNN特征
        features = self.cnn(images)
        features = features.view(features.size(0), -1)
        
        # 映射到嵌入空间
        features = self.fc(features)
        
        # L2标准化
        features = nn.functional.normalize(features, p=2, dim=1)
        
        return features


class TextEncoder(nn.Module):
    """文本编码器"""
    
    def __init__(self, vocab_size, word_dim, embed_size, num_layers=1):
        super(TextEncoder, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, word_dim)
        
        # GRU层
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.embed.weight)
    
    def forward(self, captions, lengths):
        """前向传播"""
        # 词嵌入
        embedded = self.embed(captions)
        
        # 打包序列
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # GRU前向传播
        _, hidden = self.rnn(packed)
        
        # 取最后一层的隐状态
        features = hidden[-1]
        
        # L2标准化
        features = nn.functional.normalize(features, p=2, dim=1)
        
        return features


class VSEModel(nn.Module):
    """VSE++模型"""
    
    def __init__(self, opt):
        super(VSEModel, self).__init__()
        
        # 图像编码器
        self.img_enc = ImageEncoder(
            embed_size=opt.embed_size,
            cnn_type=getattr(opt, 'cnn_type', 'resnet152'),
            finetune=getattr(opt, 'finetune', True)
        )
        
        # 文本编码器 (使用简化的词汇表)
        vocab_size = getattr(opt, 'vocab_size', 20000)
        word_dim = getattr(opt, 'word_dim', 300)
        
        self.txt_enc = TextEncoder(
            vocab_size=vocab_size,
            word_dim=word_dim,
            embed_size=opt.embed_size
        )
        
        # 损失函数相关
        self.criterion = nn.TripletMarginLoss(
            margin=getattr(opt, 'margin', 0.2)
        )
    
    def forward_emb(self, images, captions, lengths):
        """前向传播得到嵌入"""
        # 图像嵌入
        img_emb = self.img_enc(images)
        
        # 文本嵌入
        cap_emb = self.txt_enc(captions, lengths)
        
        return img_emb, cap_emb
    
    def forward_loss(self, img_emb, cap_emb):
        """计算损失"""
        scores = torch.mm(img_emb, cap_emb.t())
        diagonal = scores.diag().view(-1, 1)
        
        # 图像到文本损失
        d1 = diagonal.expand_as(scores)
        cost_s = (0.2 + scores - d1).clamp(min=0)
        
        # 文本到图像损失
        d2 = diagonal.t().expand_as(scores)
        cost_im = (0.2 + scores - d2).clamp(min=0)
        
        # 清除对角线
        mask = torch.eye(scores.size(0)) > 0.5
        if torch.cuda.is_available():
            mask = mask.cuda()
        
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        
        # 最大违反策略
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
        
        return cost_s.mean() + cost_im.mean()
    
    def forward(self, images, captions, lengths):
        """完整的前向传播"""
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)
        loss = self.forward_loss(img_emb, cap_emb)
        return loss 