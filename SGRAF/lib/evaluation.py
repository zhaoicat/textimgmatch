"""评估工具函数"""

import numpy as np
from collections import OrderedDict
import time


class AverageMeter(object):
    """计算并存储平均值和当前值"""
    
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
    """收集和记录训练日志"""
    
    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=1):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' {:.4f} ({:.4f})'.format(v.val, v.avg)
        return s


def cosine_similarity(x, y, dim=1, eps=1e-8):
    """计算余弦相似度"""
    w12 = torch.sum(x * y, dim)
    w1 = torch.norm(x, 2, dim)
    w2 = torch.norm(y, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def compute_sim(images, captions, measure='cosine'):
    """计算图像和文本的相似度矩阵"""
    if measure == 'cosine':
        # 标准化特征
        images = images / np.linalg.norm(images, axis=1, keepdims=True)
        captions = captions / np.linalg.norm(captions, axis=1, keepdims=True)
        
        # 计算余弦相似度
        similarities = np.dot(images, captions.T)
    elif measure == 'euclidean':
        # 欧几里得距离（转换为相似度）
        similarities = -np.linalg.norm(
            images[:, np.newaxis] - captions[np.newaxis, :], axis=2
        )
    else:
        raise ValueError(f"Unknown similarity measure: {measure}")
    
    return similarities


def i2t(images, captions, measure='cosine', return_ranks=False):
    """
    图像到文本检索
    Images->Text (Image Annotation)
    Images: (N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = images.shape[0]
    
    # 计算相似度
    sims = compute_sim(images, captions, measure)
    
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    
    # 检查数据格式，如果不是5倍关系，采用1:1匹配
    if captions.shape[0] == images.shape[0]:
        # 1:1匹配情况
        for index in range(npts):
            inds = np.argsort(sims[index])[::-1]
            rank = np.where(inds == index)[0]
            if len(rank) > 0:
                ranks[index] = rank[0]
            else:
                ranks[index] = npts  # 最坏情况
            top1[index] = inds[0]
    else:
        # 原来的5:1匹配情况
        for index in range(npts):
            inds = np.argsort(sims[index])[::-1]
            
            # 分数最高的检索结果
            rank = 1e20
            for i in range(5*index, min(5*index + 5, captions.shape[0])):
                tmp_rank = np.where(inds == i)[0]
                if len(tmp_rank) > 0:
                    tmp = tmp_rank[0]
                    if tmp < rank:
                        rank = tmp
            if rank == 1e20:
                rank = npts
            ranks[index] = rank
            top1[index] = inds[0]

    # 计算metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, measure='cosine', return_ranks=False):
    """
    文本到图像检索
    Text->Images (Image Search)
    Images: (N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = images.shape[0]
    
    # 计算相似度
    sims = compute_sim(images, captions, measure)
    
    # --> (5N(caption), N(image))
    sims = sims.T
    
    # 检查数据格式
    if captions.shape[0] == images.shape[0]:
        # 1:1匹配情况
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
        
        for index in range(npts):
            inds = np.argsort(sims[index])[::-1]
            rank = np.where(inds == index)[0]
            if len(rank) > 0:
                ranks[index] = rank[0]
            else:
                ranks[index] = npts
            top1[index] = inds[0]
    else:
        # 原来的5:1匹配情况
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
        
        for index in range(npts):
            for i in range(min(5, captions.shape[0] - 5*index)):
                if 5 * index + i < sims.shape[0]:
                    inds = np.argsort(sims[5 * index + i])[::-1]
                    rank_pos = np.where(inds == index)[0]
                    if len(rank_pos) > 0:
                        ranks[5 * index + i] = rank_pos[0]
                    else:
                        ranks[5 * index + i] = npts
                    top1[5 * index + i] = inds[0]

    # 计算metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def encode_data(model, data_loader, log_step=10, logging=print):
    """编码数据集"""
    # 切换到评估模式
    model.eval()
    
    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
    
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        if i % log_step == 0:
            logging('Computing results... step: %d/%d'%(i, len(data_loader)))
            
        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(images, captions, lengths)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
        
        del images, captions
    
    return img_embs, cap_embs


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    评估模型的检索性能
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    
    print(opt)
    
    # load vocabulary used by the model
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)

    # construct model
    model = VSEModel(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0], cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        
        start = time.time()
        if opt.cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, model, opt, shard_size=128)
        elif opt.cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, model, opt, shard_size=128)
        else:
            sims = shard_xattn(img_embs, cap_embs, model, opt, shard_size=128)
        end = time.time()
        print("calculate similarity time:", end-start)

        r, rt = i2t(img_embs, cap_embs, sims, return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 1000:(i + 1) * 1000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            
            start = time.time()
            if opt.cross_attn == 't2i':
                sims = shard_xattn_t2i(img_embs_shard, cap_embs_shard, model, opt, shard_size=128)
            elif opt.cross_attn == 'i2t':
                sims = shard_xattn_i2t(img_embs_shard, cap_embs_shard, model, opt, shard_size=128)
            else:
                sims = shard_xattn(img_embs_shard, cap_embs_shard, model, opt, shard_size=128)
            end = time.time()
            print("calculate similarity time:", end-start)

            r, rt0 = i2t(img_embs_shard, cap_embs_shard, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, cap_embs_shard, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[8])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[9])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10]) 