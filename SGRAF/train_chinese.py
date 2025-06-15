#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本 - 适配中文数据的SGRAF模型训练
"""

import os
import time
import shutil
import pickle

import torch
import numpy as np

import data_chinese as data
from vocab import Vocabulary  
from model import SGRAF
from evaluation import evalrank
import logging
import tensorboard_logger as tb_logger

import argparse


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/Users/gszhao/code/小红书/图文匹配/SGRAF/data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='tk_precomp',
                        help='dataset name')
    parser.add_argument('--vocab_path', default='/Users/gszhao/code/小红书/图文匹配/SGRAF/vocab/tk_precomp_vocab.pkl',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', default='/Users/gszhao/code/小红书/图文匹配/SGRAF/runs/tk_SGR/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='/Users/gszhao/code/小红书/图文匹配/SGRAF/runs/tk_SGR/log',
                        help='Path to save Tensorboard log.')
    
    # 训练参数
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')
    
    # 模型参数
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', default='SGR', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')

    opt = parser.parse_args()
    print(opt)

    # 创建保存目录
    os.makedirs(os.path.dirname(opt.model_name), exist_ok=True)
    os.makedirs(os.path.dirname(opt.logger_name), exist_ok=True)

    # 设置日志
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # 加载词汇表
    print(f"加载词汇表：{opt.vocab_path}")
    with open(opt.vocab_path, 'rb') as f:
        vocab_dict = pickle.load(f)
    
    # 创建词汇表对象
    vocab = SimpleVocab(vocab_dict)
    opt.vocab_size = len(vocab_dict)

    # 加载数据
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # 构建模型
    model = SGRAF(opt)

    # 训练模型
    train(opt, train_loader, val_loader, model, vocab)


class SimpleVocab:
    """简单的词汇表类"""
    def __init__(self, vocab_dict):
        self.stoi = vocab_dict  # string to index
        self.itos = {v: k for k, v in vocab_dict.items()}  # index to string
        
    def __call__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])
    
    def __len__(self):
        return len(self.stoi)


def train(opt, train_loader, val_loader, model, vocab):
    """训练函数"""
    # 训练模式
    best_rsum = 0
    
    for epoch in range(opt.num_epochs):
        print(f'Epoch [{epoch}/{opt.num_epochs}]')
        
        # 调整学习率
        adjust_learning_rate(opt, model.optimizer, epoch)
        
        # 训练一个epoch
        train_epoch(opt, train_loader, model, epoch)
        
        # 验证
        rsum = validate(opt, val_loader, model)
        
        # 记住最好的R@K sum并保存checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        
        if not os.path.exists(opt.model_name):
            os.makedirs(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train_epoch(opt, train_loader, model, epoch):
    """训练一个epoch"""
    # 切换到训练模式
    model.train_start()
    
    # 平均损失记录器
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    
    end = time.time()
    for i, train_data in enumerate(train_loader):
        # 计算数据加载时间
        data_time.update(time.time() - end)
        
        # 前向传播
        model.train_emb(*train_data)
        
        # 记录日志
        train_logger.update(model.logger)
        batch_time.update(time.time() - end)
        end = time.time()
        
        # 打印日志
        if i % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, elog=str(train_logger)))
        
        # 验证
        if i % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    """验证函数"""
    # 切换到验证模式
    model.val_start()
    
    # 计算图像和文本的嵌入
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)
    
    # 计算召回率指标
    (r1, r5, r10, medr, meanr) = evalrank(img_embs, cap_embs, cap_lens)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))
    
    # 计算文本到图像的召回率
    (r1i, r5i, r10i, medri, meanri) = evalrank(cap_embs, img_embs, cap_lens)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri))
    
    # 计算总分
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logging.info('Current rsum is {}'.format(currscore))
    
    # 记录到tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)
    
    return currscore


def encode_data(model, data_loader, log_step, logging):
    """编码数据"""
    # 切换到验证模式
    model.val_start()
    
    # 初始化
    img_embs = None
    cap_embs = None
    cap_lens = None
    
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
    
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # 计算嵌入
        img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        
        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        
        # 保存嵌入
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()
        
        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]
        
        if i % log_step == 0:
            logging('Computing results... step {} of {}'.format(i, len(data_loader)))
    
    return img_embs, cap_embs, cap_lens


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    """保存checkpoint"""
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """设置学习率为初始LR衰减"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """记录和更新平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LogCollector(object):
    """收集日志的工具类"""
    def __init__(self):
        self.meters = {}

    def update(self, k_v_pairs):
        for k, v in k_v_pairs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)

    def __str__(self):
        log_str = ''
        for k, v in self.meters.items():
            log_str += ' {}: {:.4f}'.format(k, v.avg)
        return log_str


if __name__ == '__main__':
    main() 