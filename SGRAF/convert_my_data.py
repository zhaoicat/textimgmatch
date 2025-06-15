#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据转换脚本 - 将用户的数据集转换为SGRAF需要的格式
"""

import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

def extract_image_features(image_path, model, transform, device):
    """
    使用预训练的ResNet提取图像特征
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(image_tensor)
        
        return features.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def prepare_feature_extractor():
    """
    准备特征提取器
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用预训练的ResNet-152
    model = models.resnet152(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉最后的分类层
    model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform, device

def convert_dataset(data_info, output_dir, split_ratios={'train': 0.8, 'dev': 0.1, 'test': 0.1}):
    """
    转换数据集为SGRAF格式
    
    Args:
        data_info: 包含图文对信息的列表或字典
        output_dir: 输出目录
        split_ratios: 数据集划分比例
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备特征提取器
    print("准备特征提取器...")
    model, transform, device = prepare_feature_extractor()
    
    # 数据集划分
    total_samples = len(data_info)
    train_end = int(total_samples * split_ratios['train'])
    dev_end = train_end + int(total_samples * split_ratios['dev'])
    
    splits = {
        'train': data_info[:train_end],
        'dev': data_info[train_end:dev_end],
        'test': data_info[dev_end:]
    }
    
    for split_name, split_data in splits.items():
        print(f"处理 {split_name} 集，共 {len(split_data)} 个样本...")
        
        captions = []
        features = []
        
        for item in tqdm(split_data):
            # 假设每个item包含 'image_path' 和 'caption'
            image_path = item['image_path']
            caption = item['caption']
            
            # 提取图像特征
            feature = extract_image_features(image_path, model, transform, device)
            if feature is not None:
                features.append(feature)
                captions.append(caption.encode('utf-8'))
        
        # 保存文本文件
        caps_file = os.path.join(output_dir, f'{split_name}_caps.txt')
        with open(caps_file, 'wb') as f:
            for caption in captions:
                f.write(caption + b'\n')
        
        # 保存特征文件
        features_array = np.array(features)
        ims_file = os.path.join(output_dir, f'{split_name}_ims.npy')
        np.save(ims_file, features_array)
        
        print(f"{split_name} 集处理完成：{len(captions)} 个样本")
        print(f"特征形状：{features_array.shape}")

def load_your_dataset(data_path):
    """
    加载你的数据集 - 这个函数需要根据你的实际数据格式来修改
    
    Args:
        data_path: 你的数据集路径
    
    Returns:
        data_info: 包含图文对信息的列表
        格式：[{'image_path': '图片路径', 'caption': '文本描述'}, ...]
    """
    
    # 示例1: 如果你的数据是JSON格式
    # with open(os.path.join(data_path, 'annotations.json'), 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # return data
    
    # 示例2: 如果你的数据是CSV格式
    # import pandas as pd
    # df = pd.read_csv(os.path.join(data_path, 'data.csv'))
    # data_info = []
    # for _, row in df.iterrows():
    #     data_info.append({
    #         'image_path': os.path.join(data_path, 'images', row['image_file']),
    #         'caption': row['caption']
    #     })
    # return data_info
    
    # 示例3: 如果图片和文本分别在不同文件夹
    # images_dir = os.path.join(data_path, 'images')
    # captions_dir = os.path.join(data_path, 'captions')
    # data_info = []
    # for img_file in os.listdir(images_dir):
    #     if img_file.endswith(('.jpg', '.png', '.jpeg')):
    #         img_path = os.path.join(images_dir, img_file)
    #         cap_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    #         cap_path = os.path.join(captions_dir, cap_file)
    #         if os.path.exists(cap_path):
    #             with open(cap_path, 'r', encoding='utf-8') as f:
    #                 caption = f.read().strip()
    #             data_info.append({
    #                 'image_path': img_path,
    #                 'caption': caption
    #             })
    # return data_info
    
    # 请根据你的实际数据格式修改这里
    print("请修改 load_your_dataset 函数以适配你的数据格式")
    return []

if __name__ == "__main__":
    # 使用方法
    # 1. 修改 load_your_dataset 函数来加载你的数据
    # 2. 设置输入和输出路径
    # 3. 运行脚本
    
    input_data_path = "你的数据集路径"  # 修改为你的数据集路径
    output_data_path = "SGRAF/data/my_dataset"  # 输出路径
    
    # 加载数据
    data_info = load_your_dataset(input_data_path)
    
    if len(data_info) > 0:
        # 转换数据集
        convert_dataset(data_info, output_data_path)
        print("数据转换完成！")
    else:
        print("请先修改 load_your_dataset 函数来加载你的数据集") 