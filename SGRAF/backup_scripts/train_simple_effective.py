#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import shutil
import pickle
import logging
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import data
from model import SGRAF
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores

def main():
    # 超参数设置
    opt = argparse.Namespace()
    opt.data_path = './data/'
    opt.data_name = 'tk_precomp'
    opt.vocab_path = './vocab/'
    opt.margin = 0.2
    opt.num_epochs = 80
    opt.batch_size = 128
    opt.word_dim = 300
    opt.embed_size = 1024
    opt.grad_clip = 2.0
    opt.crop_size = 224
    opt.num_layers = 1
    opt.learning_rate = 0.0002
    opt.lr_update = 15
    opt.workers = 10
    opt.log_step = 10
    opt.val_step = 500
    opt.logger_name = './runs/simple_effective'
    opt.model_name = './runs/simple_effective'
    opt.resume = ''
    opt.max_violation = True
    opt.img_dim = 2048
    opt.measure = 'cosine'
    opt.use_abs = False
    opt.no_imgnorm = False
    opt.reset_train = True
    opt.raw_feature_norm = "clipped_l2norm"
    opt.agg_func = "LogSumExp"
    opt.cross_attn = "t2i"
    opt.precomp_enc_type = "basic"
    opt.bi_gru = False
    opt.lambda_lse = 6.0
    opt.lambda_softmax = 9.0
    opt.no_txtnorm = False
    opt.sim_dim = 256
    opt.module_name = 'SGR'
    opt.sgr_step = 3
    
    print("=== 简单有效训练 - 专注基础优化 ===")
    print(f"基线: 37.47%")
    print(f"目标: 40-42%")
    print(f"策略: 学习率调优+数据增强+损失函数改进")
    
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger = LogCollector()

    # 加载词汇表
    with open(os.path.join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        vocab = pickle.load(f)
    opt.vocab_size = len(vocab)
    print(f"词汇表大小: {opt.vocab_size}")

    # 构建数据加载器
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")

    # 构建模型
    model = SGRAF(opt)

    # 使用GPU
    if torch.cuda.is_available():
        print("使用GPU")
    else:
        print("使用CPU")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # 训练
    print(f"开始训练 {opt.num_epochs} epochs...")
    
    best_rsum = 0
    best_r1 = 0
    
    for epoch in range(opt.num_epochs):
        print(f"\nEpoch {epoch+1}/{opt.num_epochs}")
        print("-" * 50)
        
        # 调整学习率
        adjust_learning_rate(opt, model.optimizer, epoch)
        
        # 训练一个epoch
        train_epoch(model, train_loader, epoch, opt)
        
        # 验证
        print("验证中...")
        rsum, r1_avg = validate(model, val_loader, opt)
        
        # 保存最佳模型
        is_best = rsum > best_rsum
        if is_best:
            best_rsum = rsum
            best_r1 = r1_avg
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')
            print(f"💾 最佳模型 - R-sum: {rsum:.2f}, 平均R@1: {r1_avg:.2f}%")
        
        print(f"当前最佳: {best_r1:.2f}%")
        improvement = best_r1 - 37.47
        print(f"改进: {improvement:+.2f}%")
        print(f"距离42%: {42 - best_r1:.2f}%")
        
        # 早停检查
        if epoch > 30 and improvement < -2:
            print("性能下降过多，早停")
            break

    print("\n" + "=" * 50)
    print("简单有效训练完成!")
    print(f"基线: 37.47%")
    print(f"结果: {best_r1:.2f}%")
    print(f"改进: {improvement:+.2f}%")
    if best_r1 >= 40:
        print("🎉 成功突破40%!")
    else:
        print(f"继续努力，距离40%还差: {40 - best_r1:.2f}%")

def train_epoch(model, train_loader, epoch, opt):
    # 训练模式
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # 训练模式
        model.train_start()
        
        # 数据加载时间
        data_time.update(time.time() - end)

        # 设置logger
        model.logger = train_logger

        # 更新模型
        model.train_emb(*train_data)

        # 记录时间
        batch_time.update(time.time() - end)
        end = time.time()

        # 打印训练状态
        if model.Eiters % opt.log_step == 0:
            print(f'Epoch [{epoch}][{i}/{len(train_loader)}] '
                  f'Time {batch_time.val:.3f} '
                  f'{str(model.logger)}')

def validate(model, val_loader, opt):
    # 验证模式
    model.eval()
    
    with torch.no_grad():
        # 编码所有图像和文本
        img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)
        
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

        start = time.time()
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=128)
        end = time.time()
        print("计算相似度矩阵时间: {:.3f}s".format(end-start))

        # 文本到图像检索
        (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
        print("Text to Image: R@1: {:.2f}%, R@5: {:.2f}%, R@10: {:.2f}%".format(r1, r5, r10))
        
        # 图像到文本检索
        (r1i, r5i, r10i, medri, meanri) = t2i(img_embs, cap_embs, cap_lens, sims)
        print("Image to Text: R@1: {:.2f}%, R@5: {:.2f}%, R@10: {:.2f}%".format(r1i, r5i, r10i))
        
        # 计算总分和平均R@1
        rsum = r1 + r5 + r10 + r1i + r5i + r10i
        r1_avg = (r1 + r1i) / 2
        print("平均 R@1: {:.2f}%".format(r1_avg))
        print("R-sum: {:.2f}".format(rsum))

    return rsum, r1_avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # 确保目录存在
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # 尝试保存
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('模型保存 {} 失败, 剩余尝试次数 {}'.format(filename, tries))
        if not tries:
            raise error

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main() 