#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TK数据集转换脚本 - 将唐卡图文数据转换为SGRAF需要的格式
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import random
import jieba  # 中文分词

def load_tk_dataset(data_dir):
    """
    加载TK数据集
    """
    images_dir = os.path.join(data_dir, 'newtrain')
    text_file = os.path.join(data_dir, 'all_wenben.txt')
    
    data_pairs = []
    
    # 读取文本文件
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                img_name, caption = line.split('\t', 1)
                img_path = os.path.join(images_dir, img_name)
                
                # 检查图片是否存在
                if os.path.exists(img_path):
                    data_pairs.append({
                        'image_path': img_path,
                        'caption': caption.strip(),
                        'image_name': img_name
                    })
                else:
                    print(f"警告：图片不存在 {img_path}")
    
    print(f"成功加载 {len(data_pairs)} 个图文对")
    return data_pairs

def prepare_feature_extractor():
    """
    准备特征提取器
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 使用预训练的ResNet-152
    model = models.resnet152(pretrained=True)
    # 去掉最后的分类层，保留特征提取部分
    model = torch.nn.Sequential(*list(model.children())[:-1])
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

def extract_image_features(image_path, model, transform, device):
    """
    提取单张图像的特征
    """
    try:
        # 打开并预处理图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            features = model(image_tensor)
        
        # 将特征reshape为1维
        features = features.view(features.size(0), -1)
        return features.cpu().numpy().squeeze()
    
    except Exception as e:
        print(f"处理图片出错 {image_path}: {e}")
        return None

def preprocess_text(text):
    """
    预处理中文文本
    """
    # 使用jieba进行中文分词
    words = jieba.lcut(text)
    # 去除空白和标点
    words = [word.strip() for word in words if word.strip() and len(word.strip()) > 1]
    return ' '.join(words)

def convert_tk_dataset(data_dir, output_dir, split_ratios={'train': 0.8, 'dev': 0.1, 'test': 0.1}):
    """
    转换TK数据集为SGRAF格式
    """
    print("开始转换TK数据集...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    data_pairs = load_tk_dataset(data_dir)
    
    if len(data_pairs) == 0:
        print("错误：没有找到有效的数据对")
        return
    
    # 随机打乱数据
    random.shuffle(data_pairs)
    
    # 准备特征提取器
    print("准备特征提取器...")
    model, transform, device = prepare_feature_extractor()
    
    # 数据集划分
    total_samples = len(data_pairs)
    train_end = int(total_samples * split_ratios['train'])
    dev_end = train_end + int(total_samples * split_ratios['dev'])
    
    splits = {
        'train': data_pairs[:train_end],
        'dev': data_pairs[train_end:dev_end],
        'test': data_pairs[dev_end:]
    }
    
    print(f"数据集划分：")
    print(f"训练集：{len(splits['train'])} 个样本")
    print(f"验证集：{len(splits['dev'])} 个样本")
    print(f"测试集：{len(splits['test'])} 个样本")
    
    # 处理每个数据分割
    for split_name, split_data in splits.items():
        print(f"\n处理 {split_name} 集...")
        
        captions = []
        features = []
        processed_count = 0
        
        for item in tqdm(split_data, desc=f"处理{split_name}"):
            # 提取图像特征
            feature = extract_image_features(item['image_path'], model, transform, device)
            
            if feature is not None:
                # 预处理文本
                processed_caption = preprocess_text(item['caption'])
                
                features.append(feature)
                captions.append(processed_caption.encode('utf-8'))
                processed_count += 1
        
        if processed_count > 0:
            # 保存文本文件
            caps_file = os.path.join(output_dir, f'{split_name}_caps.txt')
            with open(caps_file, 'wb') as f:
                for caption in captions:
                    f.write(caption + b'\n')
            
            # 保存特征文件
            features_array = np.array(features)
            ims_file = os.path.join(output_dir, f'{split_name}_ims.npy')
            np.save(ims_file, features_array)
            
            print(f"{split_name} 集处理完成：")
            print(f"  - 成功处理：{processed_count} 个样本")
            print(f"  - 特征形状：{features_array.shape}")
            print(f"  - 文件保存到：{caps_file}, {ims_file}")
        else:
            print(f"警告：{split_name} 集没有有效样本")

if __name__ == "__main__":
    # 设置路径
    tk_data_dir = "/Users/gszhao/code/小红书/图文匹配/SGRAF/data/tk"
    output_dir = "/Users/gszhao/code/小红书/图文匹配/SGRAF/data/tk_precomp"
    
    # 转换数据集
    convert_tk_dataset(tk_data_dir, output_dir)
    print("\n数据转换完成！")
    print(f"转换后的数据保存在：{output_dir}")
    print("\n接下来可以：")
    print("1. 修改 opts.py 中的数据路径")
    print("2. 构建词汇表")
    print("3. 开始训练模型") 