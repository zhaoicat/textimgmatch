"""Data provider for Chinese text"""

import torch
import torch.utils.data as data

import os
import jieba
import numpy as np


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Modified for Chinese text processing
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # load the raw captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())

        # load the image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = min(5000, self.length)

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = int(index // self.im_div)
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # convert caption (string) to word ids using jieba for Chinese
        tokens = jieba.lcut(caption)
        caption_ids = []
        
        # 处理字典格式的词汇表
        if isinstance(vocab, dict):
            caption_ids.append(vocab.get('<start>', 0))
            for token in tokens:  
                if token.strip():  # skip empty tokens
                    if token in vocab:
                        caption_ids.append(vocab[token])
                    else:
                        caption_ids.append(vocab.get('<unk>', 3))
            caption_ids.append(vocab.get('<end>', 2))
        else:
            # 原来的对象格式处理
            caption_ids.append(vocab('<start>'))
            for token in tokens:  
                if token.strip():  # skip empty tokens
                    if hasattr(vocab, 'stoi') and token in vocab.stoi:
                        caption_ids.append(vocab.stoi[token])
                    else:
                        caption_ids.append(vocab('<unk>') if callable(vocab) else vocab.get('<unk>', 3))
            caption_ids.append(vocab('<end>'))
            
        target = torch.Tensor(caption_ids)

        return image, target, index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (2048,).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 1D tensor to 2D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    100, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     100, False, workers)
    return test_loader


def get_tokenizer():
    """返回中文分词器"""
    return jieba.lcut 