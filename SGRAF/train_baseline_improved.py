#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import shutil
import pickle
import logging
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import data
from model import SGRAF
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # 超参数设置 - 基于原始配置的改进
    opt = argparse.Namespace()
    opt.data_path = './data/'
    opt.data_name = 'tk_precomp'
    opt.vocab_path = './vocab/'
    opt.margin = 0.2
    opt.num_epochs = 60  # 增加训练轮数
    opt.batch_size = 128
    opt.word_dim = 300
    opt.embed_size = 1024
    opt.grad_clip = 2.0
    opt.num_layers = 1
    opt.learning_rate = 0.0002  # 稍微降低学习率
    opt.lr_update = 15  # 更频繁的学习率衰减
    opt.workers = 2
    opt.log_step = 10
    opt.val_step = 500
    opt.logger_name = './logs/baseline_improved'
    opt.model_name = './models/baseline_improved'
    opt.resume = ''
    opt.max_violation = False
    opt.img_dim = 2048
    opt.no_imgnorm = False
    opt.no_txtnorm = False
    opt.raw_feature_norm = "clipped_l2norm"
    opt.agg_func = "LogSumExp"
    opt.cross_attn = "t2i"
    opt.precomp_enc_type = "basic"
    opt.bi_gru = False
    opt.lambda_lse = 6.0
    opt.lambda_softmax = 9.0
    opt.sim_dim = 256
    opt.module_name = 'SGR'
    opt.sgr_step = 3

    print("=== 基线改进训练 ===")
    print("基线: 37.47%")
    print("目标: 40-42%")
    print("策略: 优化学习率调度+增加训练轮数")
    
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
    
    # 训练
    best_rsum = 0
    best_r1 = 0

    for epoch in range(opt.num_epochs):
        print(f"\nEpoch {epoch+1}/{opt.num_epochs}")
        print("-" * 50)
        
        # 调整学习率
        adjust_learning_rate(opt, model.optimizer, epoch)
        
        # 训练一个epoch
        train(opt, train_loader, model, epoch, val_loader)
        
        # 验证
        r_sum = validate(opt, val_loader, model)
        
        # 保存最佳模型
        is_best = r_sum > best_rsum
        if is_best:
            best_rsum = r_sum
            # 计算平均R@1
            img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
            sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
            (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
            (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
            best_r1 = (r1 + r1i) / 2
            
            if not os.path.exists(opt.model_name):
                os.makedirs(opt.model_name)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')
            
            print(f"💾 最佳模型 - R-sum: {r_sum:.2f}, 平均R@1: {best_r1:.2f}%")
        
        improvement = best_r1 - 37.47
        print(f"当前最佳: {best_r1:.2f}%")
        print(f"改进: {improvement:+.2f}%")
        if best_r1 >= 40:
            print("🎉 突破40%！")
        elif best_r1 >= 38:
            print("📈 接近目标")
        else:
            print("继续努力")

def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            print(f'Epoch [{epoch}][{i}/{len(train_loader)}] '
                  f'Time {batch_time.val:.3f} '
                  f'{str(model.logger)}')

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

    # clear duplicate 5*images and keep 1*images
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    return r_sum

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
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
        print('model save {} failed, remaining {} trials'.format(filename, tries))
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