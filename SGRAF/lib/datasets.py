"""Dataset for raw images and captions"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import json


class RawImageDataset(data.Dataset):
    """
    Dataset for loading raw images and captions
    """

    def __init__(self, data_path, data_split, tokenizer, opt=None):
        self.data_path = data_path
        self.data_split = data_split
        self.tokenizer = tokenizer
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """加载图像和文本数据"""
        # 构建数据路径
        data_file = os.path.join(self.data_path, f'{self.data_split}_caps.txt')
        
        self.captions = []
        self.image_paths = []
        
        if os.path.exists(data_file):
            # 从文本文件加载
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                caption = line.strip()
                if caption:
                    self.captions.append(caption)
                    # 假设图像文件命名为 image_0.jpg, image_1.jpg, ...
                    img_path = os.path.join(self.data_path, f'image_{i}.jpg')
                    self.image_paths.append(img_path)
        else:
            # 示例数据用于测试
            print(f"数据文件 {data_file} 不存在，使用示例数据")
            for i in range(100):  # 创建100个示例
                self.captions.append(f"这是第{i}张图片的描述")
                # 创建一个虚拟图像路径
                img_path = os.path.join(self.data_path, f'dummy_image_{i}.jpg')
                self.image_paths.append(img_path)
    
    def __getitem__(self, index):
        # 加载图像
        img_path = self.image_paths[index]
        
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # 创建一个虚拟图像用于测试
            image = Image.new('RGB', (224, 224), color='white')
        
        image = self.transform(image)
        
        # 处理文本
        caption = self.captions[index]
        tokens = self.tokenizer(caption) if callable(self.tokenizer) else caption.split()
        
        # 转换为token id (简化处理)
        token_ids = [hash(token) % 10000 for token in tokens]  # 简化的词汇映射
        target = torch.tensor(token_ids)
        
        length = len(tokens)
        
        return image, target, length, index
    
    def __len__(self):
        return len(self.captions)
    
    @staticmethod
    def collate_fn(data):
        """
        自定义的collate function
        """
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, lengths, ids = zip(*data)
        
        # 合并图像
        images = torch.stack(images, 0)
        
        # 处理不同长度的caption
        max_len = max(lengths)
        targets = torch.zeros(len(captions), max_len).long()
        
        for i, cap in enumerate(captions):
            end = lengths[i]
            if end > 0:
                targets[i, :end] = cap[:end]
        
        return images, targets, list(lengths), list(ids) 