#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V73 甲状腺结节特征CNN模型训练脚本 (EfficientNet-B0 多任务版)

v73: verify

v70: all the dataset balanced with fna/surgery_0924 and tr1-3
     all_matched_sops_ds_v3_0924.csv
     + none_single_tr13.csv
     = all_matched_sops_ds_v3_with_tr13_0926.csv

V60: OOF_p_true_threshold=0.5
V61: OOF_p_true_threshold=0.2
V64: OOF_p_true_threshold=0.1, image_index_threshold=16

v66: OOF_p_true_threshold=0.1, image_index_threshold=16
     dataset -> sop_fna_nodules_with_path_v3_with_OOF_suspect.csv
     bom_weight: 0.9, ti_rads:0.5, feature:0.1 x 5

v67: OOF_p_true_threshold=0.1, image_index_threshold=16
    dataset -> sop_fna_nodules_with_path_v3_with_OOF_suspect.csv
    dropout=0.5

v68: OOF_p_true_threshold=0.3, image_index_threshold=16
    dataset -> sop_fna_nodules_with_path_v3_with_OOF_suspect.csv
    verify -> 0809_v3
    dropout=0.5

v68: OOF_p_true_threshold=0.5, image_index_threshold=16
    dataset -> sop_fna_nodules_with_path_v3_with_OOF_suspect.csv
    verify -> 0809_v3
    dropout=0.6

V64 核心特性:
1. 多任务学习:
   - 主要任务: BOM分类 (良恶性)
   - 辅助任务: TI-RADS分类 (1-5级), 以及5个超声征象分类 (composition, echo, foci, margin, shape)
2. 统一特征映射:
   - 所有分类任务的标签和序号均由 `core/utils/all_features_mapping_numer_v4.json` 定义
   - 映射表会嵌入到模型中，便于推理时直接调用
3. 复杂的筛选策略:
   - 图像存在性检查
   - 根据OOF预测剔除疑似错标样本 (当OOF_p_true_threshold>0时: p_true < threshold & pred != true)
   - 剔除指定月份数据 (2024年8-9月)，防止数据泄露
   - 剔除TI-RADS=6的样本 (已病理证实的恶性)
   - 剔除image_index > 16的数据 (sop_uid中以'.'分隔的倒数第二段)
4. 损失权重:
   - bom_weight: 0.7
   - 其他6个辅助任务权重各0.05
5. 独立验证集:
   - 定期 (每5个epoch) 在独立的验证集上评估性能，监控泛化能力
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error
from torch.utils.tensorboard import SummaryWriter  # 临时禁用
import warnings
import matplotlib.pyplot as plt
import random
import re
from datetime import datetime
import io
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# 配置管理 - V64 配置项统一管理
# =============================================================================
CONFIG = {
    # 数据集配置
    'sop4_data': 'data/dataset_table/train/all_matched_sops_ds_v3_with_tr13_0926_with_OOF_suspect.csv',
    'verify_data': 'data/dataset_table/val/all_verify_sop_with_predictions.csv',
    'image_root': 'data/dataset_images/2nodule_images',
    'verify_root': 'data/dataset_images/2nodule_images',
    'feature_mapping_file': 'utils/all_features_mapping_numer_v4.json',
    
    # LMDB配置 (高速I/O)
    'use_lmdb': True,  # 是否使用LMDB加速数据加载
    'train_lmdb_path': 'data/dataset_lmdb/train_lmdb',
    'val_lmdb_path': 'data/dataset_lmdb/val_lmdb',
    'verify_lmdb_path': 'data/dataset_lmdb/verify_lmdb',

    # 模型配置
    'model_name': 'v75',
    'output_dir': 'data/models/nodule_feature_cnn_v75',
    'tensorboard_dir': 'runs/nodule_feature_cnn_v75',
    'dropout_rate': 0.4,

    # 训练配置
    'batch_size': 256,  # 增大batch让每卡计算量更多
    'num_epochs': 50,
    'learning_rate': 5e-5,
    'weight_decay': 1e-2,
    'early_stop_patience': 10,
    'max_grad_norm': 1.0,
    'verify_epoch_interval': 5,

    # 数据筛选配置
    'OOF_p_true_threshold': 0.2,
    'image_index_threshold': 16,
    'exclude_months': ['202408', '202409'],
    'exclude_ti_rads': [6],

    # 损失权重配置 (总和应为1.0)
    'loss_weights': {
        'bom': 0.9,
        'ti_rads': 0.05,
        'composition': 0.01,
        'echo': 0.01,
        'foci': 0.01,
        'margin': 0.01,
        'shape': 0.01
    },

    # 数据划分配置
    'test_size': 0.2,
    'random_state': 42,
    'num_workers': 8,  # 单GPU配合LMDB高速读取
    
    # 多GPU配置
    'use_multi_gpu': False,  # 单卡训练（小模型效率最高）
    'gpu_ids': [0],  # 单卡
}

# =============================================================================
# 日志记录类
# =============================================================================

class TrainingLogger:
    """训练日志记录器 - 同时输出到控制台和文件"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_buffer = []
        self.best_epoch_info = {}
        self.final_verify_info = {}

    def log(self, message):
        """记录消息到缓冲区和控制台"""
        print(message)
        self.log_buffer.append(message)

    def log_best_epoch(self, epoch, train_metrics, val_metrics):
        """记录最佳epoch信息"""
        self.best_epoch_info = {
            'epoch': epoch + 1,
            'train_metrics': train_metrics.copy(),
            'val_metrics': val_metrics.copy()
        }

    def log_final_verify(self, verify_metrics):
        """记录最终验证信息"""
        self.final_verify_info = verify_metrics.copy()

    def save_training_summary(self, config, best_auc, total_epochs):
        """保存训练总结到文件"""
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"V64 甲状腺结节特征CNN模型训练报告\n")
                f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                # 1. 模型配置信息
                f.write("📋 模型配置信息:\n")
                f.write(f"  模型版本: {config['model_name']}\n")
                f.write(f"  模型架构: EfficientNet-B0 多任务版\n")
                f.write(f"  输出目录: {config['output_dir']}\n")
                f.write(f"  训练图像根目录: {config['image_root']}\n")
                if 'verify_root' in config:
                    f.write(f"  验证图像根目录: {config['verify_root']}\n")
                else:
                    f.write(f"  验证图像根目录: {config['image_root']} (与训练相同)\n")
                f.write(f"  批次大小: {config['batch_size']}\n")
                f.write(f"  学习率: {config['learning_rate']}\n")
                f.write(f"  权重衰减: {config['weight_decay']}\n")
                f.write(f"  早停轮数: {config['early_stop_patience']}\n")
                f.write(f"  Dropout率: {config['dropout_rate']}\n\n")

                # 2. 数据筛选配置
                f.write("🔍 数据筛选配置:\n")
                oof_threshold = config['OOF_p_true_threshold']
                if oof_threshold == 0:
                    f.write(f"  OOF_p_true_threshold: {oof_threshold} (跳过OOF筛选)\n")
                else:
                    f.write(f"  OOF_p_true_threshold: {oof_threshold}\n")
                f.write(f"  image_index_threshold: {config['image_index_threshold']}\n")
                f.write(f"  排除月份: {config['exclude_months']}\n")
                f.write(f"  排除TI-RADS: {config['exclude_ti_rads']}\n\n")

                # 3. 损失权重配置
                f.write("⚖️ 损失权重配置:\n")
                for task, weight in config['loss_weights'].items():
                    f.write(f"  {task.capitalize():<12}: {weight}\n")
                f.write("\n")

                # 4. 训练结果概要
                f.write("🎯 训练结果概要:\n")
                f.write(f"  总训练轮数: {total_epochs}\n")
                f.write(f"  最佳验证AUC: {best_auc:.4f}\n")
                if self.best_epoch_info:
                    f.write(f"  最佳epoch: {self.best_epoch_info['epoch']}\n")
                f.write("\n")

                # 5. 最佳epoch详细指标
                if self.best_epoch_info:
                    f.write("🏆 最佳Epoch详细指标:\n")
                    f.write(f"  Epoch: {self.best_epoch_info['epoch']}\n")

                    train_metrics = self.best_epoch_info['train_metrics']
                    val_metrics = self.best_epoch_info['val_metrics']

                    f.write("  训练集指标:\n")
                    f.write(f"    BOM AUC: {train_metrics.get('bom_auc', 0):.4f}\n")
                    f.write(f"    BOM Accuracy: {train_metrics.get('bom_accuracy', 0):.4f}\n")
                    f.write(f"    BOM Sensitivity: {train_metrics.get('bom_sensitivity', 0):.4f}\n")
                    f.write(f"    BOM Specificity: {train_metrics.get('bom_specificity', 0):.4f}\n")

                    f.write("  验证集指标:\n")
                    f.write(f"    BOM AUC: {val_metrics.get('bom_auc', 0):.4f}\n")
                    f.write(f"    BOM Accuracy: {val_metrics.get('bom_accuracy', 0):.4f}\n")
                    f.write(f"    BOM Sensitivity: {val_metrics.get('bom_sensitivity', 0):.4f}\n")
                    f.write(f"    BOM Specificity: {val_metrics.get('bom_specificity', 0):.4f}\n")

                    # 辅助任务准确率
                    f.write("  辅助任务准确率:\n")
                    aux_tasks = ['ti_rads', 'composition', 'echo', 'foci', 'margin', 'shape']
                    for task in aux_tasks:
                        train_acc = train_metrics.get(f'{task}_accuracy', 0)
                        val_acc = val_metrics.get(f'{task}_accuracy', 0)
                        f.write(f"    {task.capitalize():<12} - 训练: {train_acc:.3f}, 验证: {val_acc:.3f}\n")
                    f.write("\n")

                # 6. 独立验证集最终结果
                if self.final_verify_info:
                    f.write("🧪 独立验证集最终结果:\n")
                    f.write(f"  数据集: {config['verify_data']}\n")
                    f.write(f"  BOM AUC: {self.final_verify_info.get('bom_auc', 0):.4f}\n")
                    f.write(f"  BOM Accuracy: {self.final_verify_info.get('bom_accuracy', 0):.4f}\n")
                    f.write(f"  BOM Sensitivity: {self.final_verify_info.get('bom_sensitivity', 0):.4f}\n")
                    f.write(f"  BOM Specificity: {self.final_verify_info.get('bom_specificity', 0):.4f}\n")

                    # 辅助任务准确率
                    f.write("  辅助任务准确率:\n")
                    for task in aux_tasks:
                        acc = self.final_verify_info.get(f'{task}_accuracy', 0)
                        f.write(f"    {task.capitalize():<12}: {acc:.4f}\n")
                    f.write("\n")

                # 7. 完整训练日志
                f.write("=" * 80 + "\n")
                f.write("📝 完整训练日志:\n")
                f.write("=" * 80 + "\n")
                for log_line in self.log_buffer:
                    f.write(log_line + "\n")

            print(f"📄 训练报告已保存: {self.log_file_path}")

        except Exception as e:
            print(f"❌ 保存训练报告失败: {e}")

# =============================================================================
# 核心函数
# =============================================================================

def get_device():
    """获取最佳可用设备 - CUDA优先"""
    if torch.cuda.is_available():
        # 打印可用GPU信息
        num_gpus = torch.cuda.device_count()
        print(f"🎮 检测到 {num_gpus} 个 CUDA GPU:")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_num_gpus():
    """获取可用GPU数量"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1

def load_config():
    """加载配置"""
    return CONFIG

def extract_date_from_access_no(access_no):
    """从access_no中提取年月信息 (YYYYMM)"""
    try:
        parts = access_no.split('.')
        if len(parts) >= 2:
            datetime_str = parts[1]
            if len(datetime_str) >= 6:
                year_month = datetime_str[:6]
                if len(year_month) == 6 and year_month.isdigit():
                    return year_month
    except Exception:
        pass
    return None

def extract_image_index_from_sop_uid(sop_uid):
    """从sop_uid中提取image_index (以'.'分隔的倒数第二段)"""
    try:
        parts = str(sop_uid).split('.')
        if len(parts) >= 2:
            # 倒数第二段
            image_index_str = parts[-2]
            # 尝试转换为整数
            return int(image_index_str)
    except (ValueError, IndexError):
        pass
    return None

def add_image_index_column(df):
    """为DataFrame添加image_index列（如果不存在的话）"""
    if 'image_index' not in df.columns:
        print("    - Generating image_index column from sop_uid...")
        df['image_index'] = df['sop_uid'].apply(extract_image_index_from_sop_uid)
        # 统计生成结果
        valid_count = df['image_index'].notna().sum()
        total_count = len(df)
        print(f"    - Generated image_index for {valid_count}/{total_count} records ({valid_count/total_count:.1%})")
    else:
        print("    - image_index column already exists.")
    return df

def read_csv_with_encoding(file_path, logger=None):
    """
    尝试使用不同编码读取CSV文件
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if logger:
                logger.log(f"    - Successfully loaded with {encoding} encoding.")
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Failed to read {file_path} with any of the tried encodings: {encodings}")

def apply_all_filters(df, config, df_name="DataFrame", skip_time_filter=False):
    """
    对给定的DataFrame应用所有筛选规则.
    """
    print(f"\nApplying all filters to {df_name}...")
    original_count = len(df)

    # 0. 添加image_index列（如果不存在）
    df = add_image_index_column(df)

    # 1. image_index筛选
    print("  - Rule 1: Filtering by image_index...")
    image_index_threshold = config.get('image_index_threshold', 16)
    print(f"skipping image_index filtering")
    # if 'image_index' in df.columns:
    #     # 先统计有效的image_index
    #     valid_image_index_mask = df['image_index'].notna()
    #     valid_count_before = valid_image_index_mask.sum()
    #
    #     # 对有效的image_index进行过滤
    #     image_index_mask = (df['image_index'].notna()) & (df['image_index'] > image_index_threshold)
    #     removed_count = image_index_mask.sum()
    #     df = df[~image_index_mask]
    #
    #     print(f"    - Found {valid_count_before} records with valid image_index")
    #     print(f"    - Removed {removed_count} records with image_index > {image_index_threshold}")
    # else:
    #     print(f"    - No image_index column found, skipping image_index filtering")

    # 2. 时间筛选 (可选择跳过)
    if not skip_time_filter:
        print("  - Rule 2: Filtering by date...")
        df['year_month'] = df['access_no'].apply(extract_date_from_access_no)
        time_mask = df['year_month'].isin(config['exclude_months'])
        df = df[~time_mask].drop(columns=['year_month'])
        print(f"    - Removed {time_mask.sum()} records from excluded months.")
    else:
        print("  - Rule 2: Skipping date filtering for verification data.")

    # 3. TI-RADS筛选
    print("  - Rule 3: not Filter by TI-RADS level...")
    #ti_rads_mask = df['ti_rads'].isin(config['exclude_ti_rads'])
    #df = df[~ti_rads_mask]
    #print(f"    - Removed {ti_rads_mask.sum()} records with excluded TI-RADS.")
    
    # 4. 疑似错标样本筛选 (仅对训练集应用)
    threshold = config.get('OOF_p_true_threshold', 0.5)
    if threshold > 0 and 'p_true' in df.columns:
        print("  - Rule 4: Filtering suspicious OOF samples...")
        required_cols = ['p_true', 'predicted_class', 'bom']

        # 确保列存在且为数值类型
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 仅对存在所需列的行进行操作
        valid_oof_rows = df.dropna(subset=required_cols)
        oof_mask = (valid_oof_rows['p_true'] < threshold) & (valid_oof_rows['predicted_class'] != valid_oof_rows['bom'])

        # 获取要移除的行的索引
        indices_to_remove = valid_oof_rows[oof_mask].index
        df = df.drop(indices_to_remove)
        print(f"    - Removed {len(indices_to_remove)} suspicious OOF samples (p_true < {threshold}).")
    elif threshold == 0:
        print("  - Rule 4: Skipping OOF filtering (OOF_p_true_threshold=0).")

    final_count = len(df)
    print(f"  - Filtering complete. Kept {final_count}/{original_count} records ({final_count/original_count:.2%})")
    
    return df.reset_index(drop=True)

def get_transforms_v60():
    """V60 数据增强策略"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# --- 数据集类 (V60 多任务版) ---
class NoduleFeatureDataset_V60(Dataset):
    """V60数据集类 - 支持BOM, TI-RADS, 和5个超声征象的多任务分类"""
    def __init__(self, df, image_root, feature_mapping, transform=None):
        self.df = df.copy()
        self.image_root = image_root
        self.feature_mapping = feature_mapping
        self.transform = transform
        
        self.tasks = list(feature_mapping.keys())
        
        # 为每个征象创建反向映射，用于将文本标签转换为整数
        self.label_to_int_maps = self._create_label_maps()

    def _create_label_maps(self):
        label_maps = {}
        # 'bom' 和 'ti_rads' 已经在数值映射文件中定义好了
        for task in self.tasks:
            label_maps[task] = self.feature_mapping[task]
        return label_maps

    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row['access_no'], f"{row['sop_uid']}.jpg")

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0)) # Return a black image
            print(f"image not found:{image_path}")

        if self.transform:
            image = self.transform(image)

        item = {'image': image, 'access_no': row['access_no'], 'sop_uid': row['sop_uid']}

        # 为每个任务获取标签和有效性标志
        for task in self.tasks:
            raw_val = row.get(task, np.nan)
            label_val = -1
            is_valid = 0.0

            if pd.notna(raw_val):
                # The keys in our mapping are strings. Convert raw_val to string for lookup.
                # e.g., for ti_rads, raw_val might be 1.0, we need '1'
                key = str(raw_val) if not isinstance(raw_val, str) else raw_val
                if isinstance(raw_val, float) and raw_val.is_integer():
                    key = str(int(raw_val))

                task_map = self.label_to_int_maps[task]

                if key in task_map:
                    label_val = task_map[key]
                    is_valid = 1.0

                    # Special handling for ti_rads (convert 1-5 to 0-4 for loss function)
                    if task == 'ti_rads':
                        label_val -= 1
            
            item[task] = torch.tensor(label_val, dtype=torch.long)
            item[f'{task}_valid'] = torch.tensor(is_valid, dtype=torch.float32)
            
        return item

# --- LMDB数据集类 (高速I/O版) ---
class NoduleFeatureDataset_LMDB(Dataset):
    """LMDB数据集类 - 高速I/O，支持BOM, TI-RADS, 和5个超声征象的多任务分类"""
    def __init__(self, lmdb_path, feature_mapping, transform=None, indices=None):
        import lmdb
        import pickle
        
        self.lmdb_path = lmdb_path
        self.feature_mapping = feature_mapping
        self.transform = transform
        self.tasks = list(feature_mapping.keys())
        self.label_to_int_maps = {task: feature_mapping[task] for task in self.tasks}
        
        # 延迟打开LMDB（避免pickle问题）
        self.env = None
        
        # 临时打开读取元数据
        temp_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with temp_env.begin(write=False) as txn:
            meta = pickle.loads(txn.get(b'__meta__'))
            total_samples = meta['num_samples']
        temp_env.close()
        
        # 支持索引过滤（用于训练/验证划分）
        if indices is not None:
            self.indices = list(indices)
            self.num_samples = len(self.indices)
        else:
            self.indices = list(range(total_samples))
            self.num_samples = total_samples
        
        print(f"    LMDB数据集加载完成: {self.num_samples} 样本 (总共{total_samples})")
    
    def _init_db(self):
        """在每个worker中延迟初始化LMDB连接"""
        if self.env is None:
            import lmdb
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)

    def __len__(self): 
        return self.num_samples
        
    def __getitem__(self, idx):
        import pickle
        from io import BytesIO
        
        # 延迟初始化LMDB连接
        self._init_db()
        
        # 映射到实际LMDB索引
        real_idx = self.indices[idx]
        
        # 从LMDB读取数据
        with self.env.begin(write=False) as txn:
            value = txn.get(f"{real_idx}".encode())
            if value is None:
                # fallback: 返回空数据
                return self._get_empty_item()
            data = pickle.loads(value)
        
        # 解码图像
        image_bytes = data['image']
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        item = {'image': image, 'access_no': data['access_no'], 'sop_uid': data['sop_uid']}
        
        # 为每个任务获取标签和有效性标志
        for task in self.tasks:
            raw_val = data.get(task)
            label_val = -1
            is_valid = 0.0
            
            if raw_val is not None and pd.notna(raw_val):
                key = str(raw_val) if not isinstance(raw_val, str) else raw_val
                if isinstance(raw_val, float) and raw_val == int(raw_val):
                    key = str(int(raw_val))
                
                task_map = self.label_to_int_maps[task]
                if key in task_map:
                    label_val = task_map[key]
                    is_valid = 1.0
                    if task == 'ti_rads':
                        label_val -= 1
            
            item[task] = torch.tensor(label_val, dtype=torch.long)
            item[f'{task}_valid'] = torch.tensor(is_valid, dtype=torch.float32)
        
        return item
    
    def _get_empty_item(self):
        """返回空数据项"""
        item = {
            'image': torch.zeros(3, 224, 224),
            'access_no': '',
            'sop_uid': ''
        }
        for task in self.tasks:
            item[task] = torch.tensor(-1, dtype=torch.long)
            item[f'{task}_valid'] = torch.tensor(0.0, dtype=torch.float32)
        return item

# --- V60 EfficientNet-B0 多任务模型 ---
class MultiTaskNoduleCNN_V60(nn.Module):
    """V60 EfficientNet-B0模型 - 支持7个分类任务"""
    def __init__(self, feature_mappings, dropout_rate=0.4):
        super().__init__()
        self.mappings = feature_mappings

        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone_features = 1280
        self.backbone.classifier = nn.Identity()

        self.shared_features = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.75)
        )

        # 动态创建分类头
        self.heads = nn.ModuleDict()
        for task, mapping in self.mappings.items():
            num_classes = len(mapping)
            self.heads[task] = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)

        outputs = {}
        for task, head in self.heads.items():
            outputs[task] = head(shared)

        return outputs

# --- V60损失管理器 ---
class FocusedLossManager_V60:
    """V60 损失管理器 - 为7个任务计算带权重的损失"""
    def __init__(self, loss_weights, device):
        self.loss_weights = loss_weights
        self.device = device
        self.tasks = list(loss_weights.keys())
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, outputs, batch):
        losses = {}
        total_loss = 0

        for task in self.tasks:
            # 检查是否有有效标签
            if f'{task}_valid' in batch and batch[f'{task}_valid'].sum() > 0:
                # 获取logits和labels
                logits = outputs[task]
                labels = batch[task]
                valid_mask = batch[f'{task}_valid'] > 0

                # 只对有效样本计算损失
                valid_logits = logits[valid_mask]
                valid_labels = labels[valid_mask]
                loss = self.loss_fn(valid_logits, valid_labels).mean()
                losses[task] = loss
                total_loss += self.loss_weights[task] * loss

        losses['total'] = total_loss
        return losses

# --- 训练和验证函数 ---
def train_epoch(model, loader, loss_fn, optimizer, device, config=None):
    model.train()
    total_loss = 0
    # 处理DataParallel包装
    model_mappings = model.module.mappings if isinstance(model, nn.DataParallel) else model.mappings
    metrics_calc = DetailedMetricsCalculator_V60(model_mappings)
    metrics_calc.reset()  # 确保重置

    for batch in tqdm(loader, desc="Training", leave=False):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()
        outputs = model(batch['image'])
        losses = loss_fn(outputs, batch)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'] if config else 1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        metrics_calc.update(outputs, batch)

    return total_loss / len(loader), metrics_calc.compute_metrics()

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    # 处理DataParallel包装
    model_mappings = model.module.mappings if isinstance(model, nn.DataParallel) else model.mappings
    metrics_calc = DetailedMetricsCalculator_V60(model_mappings)
    metrics_calc.reset()  # 确保重置

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch['image'])
            losses = loss_fn(outputs, batch)
            total_loss += losses['total'].item()
            metrics_calc.update(outputs, batch)

    return total_loss / len(loader), metrics_calc.compute_metrics()

# --- V60指标计算器 ---
class DetailedMetricsCalculator_V60:
    def __init__(self, mappings):
        self.mappings = mappings
        self.tasks = list(self.mappings.keys())
        self.reset()

    def reset(self):
        self.targets = {task: [] for task in self.tasks}
        self.preds = {task: [] for task in self.tasks}
        # bom需要概率用于AUC计算
        self.bom_probs = []

    def update(self, outputs, batch):
        for task in self.tasks:
            if batch[f'{task}_valid'].sum() > 0:
                valid_indices = batch[f'{task}_valid'] > 0

                self.targets[task].append(batch[task][valid_indices])
                self.preds[task].append(torch.argmax(outputs[task], dim=1)[valid_indices])

                if task == 'bom':
                    # 先应用valid_indices，再计算softmax和取第1类概率
                    bom_logits_valid = outputs['bom'][valid_indices]
                    bom_probs_valid = torch.softmax(bom_logits_valid, dim=1)[:, 1]
                    self.bom_probs.append(bom_probs_valid)


    def compute_metrics(self):
        metrics = {}
        for task in self.tasks:
            # 更严格的检查：确保有有效数据
            if not self.targets[task] or len(self.targets[task]) == 0:
                continue

            # 检查是否所有张量都非空
            valid_targets = [t for t in self.targets[task] if t.numel() > 0]
            valid_preds = [p for p in self.preds[task] if p.numel() > 0]

            if len(valid_targets) == 0 or len(valid_preds) == 0:
                continue

            targets = torch.cat(valid_targets).cpu().numpy()
            preds = torch.cat(valid_preds).cpu().numpy()

            if task == 'bom':
                valid_probs = [p for p in self.bom_probs if p.numel() > 0]
                if len(valid_probs) == 0:
                    continue

                probs = torch.cat(valid_probs).detach().cpu().numpy()

                if len(np.unique(targets)) < 2:
                    # 当只有一个类别时，无法计算AUC，设为NaN
                    metrics['bom_auc'] = float('nan')
                else:
                    metrics['bom_auc'] = roc_auc_score(targets, probs)

                tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
                metrics['bom_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['bom_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['bom_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            else:
                # 其他任务计算准确率
                accuracy = (preds == targets).mean()
                metrics[f'{task}_accuracy'] = accuracy
        return metrics

def print_results(epoch, train_metrics, val_metrics):
    print(f"\n📊 Epoch {epoch+1} Results (V64):")
    # BOM 指标 - 处理NaN值
    def format_metric(value, default=0):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:.4f}"

    train_auc = format_metric(train_metrics.get('bom_auc'))
    train_acc = format_metric(train_metrics.get('bom_accuracy'))
    train_sens = format_metric(train_metrics.get('bom_sensitivity'))
    train_spec = format_metric(train_metrics.get('bom_specificity'))

    val_auc = format_metric(val_metrics.get('bom_auc'))
    val_acc = format_metric(val_metrics.get('bom_accuracy'))
    val_sens = format_metric(val_metrics.get('bom_sensitivity'))
    val_spec = format_metric(val_metrics.get('bom_specificity'))

    print(f"  🎯 Training:   AUC={train_auc}, Acc={train_acc}, Sens={train_sens}, Spec={train_spec}")
    print(f"  🔍 Validation: AUC={val_auc}, Acc={val_acc}, Sens={val_sens}, Spec={val_spec}")

    # 辅助任务指标
    aux_tasks = [k.replace('_accuracy', '') for k in val_metrics.keys() if '_accuracy' in k and 'bom' not in k]
    for task in sorted(aux_tasks):
        train_acc = train_metrics.get(f'{task}_accuracy', 0)
        val_acc = val_metrics.get(f'{task}_accuracy', 0)
        print(f"    - {task.capitalize():<12} Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

# --- 可视化函数 (参考V37) ---
def create_image_grid(images, labels, predictions, access_nos, sop_uids, title, save_path):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(6, 6, figsize=(20, 24))  # 增加高度以容纳更多文本
    fig.suptitle(title, fontsize=16, fontweight='bold')

    def split_text_to_lines(text, max_chars_per_line=20):
        """将长文本分割成多行"""
        if len(text) <= max_chars_per_line:
            return [text]

        lines = []
        for i in range(0, len(text), max_chars_per_line):
            lines.append(text[i:i + max_chars_per_line])
        return lines

    for i in range(6):
        for j in range(6):
            ax = axes[i, j]
            idx = i * 6 + j
            if idx >= len(images):
                ax.axis('off')
                continue

            img = np.clip(images[idx].permute(1, 2, 0).cpu().numpy(), 0, 1)
            ax.imshow(img)

            bom_label = labels[idx]
            pred_val = predictions[idx]
            pred_class = 1 if pred_val > 0.5 else 0
            color = 'green' if bom_label == pred_class else 'red'

            # 左上角显示BOM标签
            ax.text(0.05, 0.95, f"BOM: {bom_label}", transform=ax.transAxes,
                   color='white', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7),
                   va='top')

            # 右上角显示预测概率
            ax.text(0.95, 0.95, f"{pred_val:.3f}", transform=ax.transAxes,
                   color=color, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8),
                   ha='right', va='top')

            # 图像下方显示完整的标识信息
            access_no = access_nos[idx]
            sop_uid = sop_uids[idx]

            # 显示access_no (缩写)
            ax.text(0.5, -0.02, f"Access: {access_no[:12]}{'...' if len(access_no) > 12 else ''}",
                   transform=ax.transAxes, color='blue', fontsize=7,
                   ha='center', va='top', fontweight='bold')

            # 分行显示完整的sop_uid
            sop_lines = split_text_to_lines(sop_uid, max_chars_per_line=25)
            for line_idx, line in enumerate(sop_lines):
                y_pos = -0.06 - (line_idx * 0.04)  # 每行间隔0.04
                ax.text(0.5, y_pos, line,
                       transform=ax.transAxes, color='darkgreen', fontsize=6,
                       ha='center', va='top', fontfamily='monospace')

            # 在右下角显示图像序号，便于快速定位
            ax.text(0.95, 0.05, f"#{idx+1}", transform=ax.transAxes,
                   color='orange', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", fc='yellow', alpha=0.6),
                   ha='right', va='bottom')

            ax.axis('off')

    # 调整布局，为底部文本留出更多空间
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    📸 Image grid saved: {save_path}")
    plt.close()

def visualize_epoch_samples(model, loader, device, epoch, output_dir, set_name="val"):
    model.eval()

    # 设置随机种子，每个epoch都不同
    random.seed(epoch * 1000 + 42)
    torch.manual_seed(epoch * 1000 + 42)

    total_batches = len(loader)
    if total_batches == 0:
        print(f"    ⚠️ {set_name} loader is empty, skipping visualization.")
        return

    # 跨batch采样，避免重复图像
    num_samples = 36
    samples_per_batch = max(1, num_samples // min(total_batches, num_samples))

    # 随机选择多个batch
    selected_batches = random.sample(range(total_batches), min(total_batches, (num_samples + samples_per_batch - 1) // samples_per_batch))

    images, labels, predictions, access_nos, sop_uids = [], [], [], [], []
    collected_samples = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx not in selected_batches or collected_samples >= num_samples:
            continue

        # 移动数据到设备
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            outputs = model(batch['image'])
            probs = torch.softmax(outputs['bom'], dim=1)[:, 1]

        # 从当前batch中采样，确保不重复access_no
        batch_size = len(batch['image'])
        remaining_samples = min(samples_per_batch, num_samples - collected_samples, batch_size)

        # 按access_no去重采样
        unique_access_indices = []
        seen_access_nos = set()

        for i in range(batch_size):
            access_no = batch['access_no'][i]
            if access_no not in seen_access_nos:
                unique_access_indices.append(i)
                seen_access_nos.add(access_no)

        # 从去重后的索引中随机选择
        if len(unique_access_indices) > 0:
            selected_indices = random.sample(unique_access_indices, min(remaining_samples, len(unique_access_indices)))

            for idx in selected_indices:
                if collected_samples >= num_samples:
                    break

                img = batch['image'][idx].cpu()
                # 反标准化
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                images.append(img)
                labels.append(batch['bom'][idx].cpu().item())
                predictions.append(probs[idx].cpu().item())
                access_nos.append(batch['access_no'][idx])
                sop_uids.append(batch['sop_uid'][idx])
                collected_samples += 1

    # 如果样本不足36张，用实际收集到的数量
    actual_samples = len(images)

    save_path = os.path.join(output_dir, f'{set_name}_samples_epoch_{epoch + 1}.png')
    create_image_grid(images, labels, predictions, access_nos, sop_uids,
                     title=f"{set_name.capitalize()} Samples - Epoch {epoch + 1} ({actual_samples} unique patients)",
                     save_path=save_path)

    print(f"    📸 可视化完成: 从{len(selected_batches)}个batch中采集{actual_samples}张图像 (去重后)")

def main():
    """主训练函数"""
    # 设置multiprocessing启动方法为spawn，解决Python 3.14兼容性问题
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 已经设置过了
    
    # CUDA环境优化设置
    if torch.cuda.is_available():
        # 启用cuDNN自动调优，针对固定输入尺寸加速卷积操作
        torch.backends.cudnn.benchmark = True
        # 启用TF32以在Ampere及更新架构上加速计算（如5090）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 设置默认CUDA设备
        torch.cuda.set_device(0)
    
    device = get_device()
    config = load_config()
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(config['tensorboard_dir'])

    # 初始化日志记录器
    log_file_path = os.path.join(output_dir, f'training_report_{config["model_name"]}.txt')
    logger = TrainingLogger(log_file_path)

    logger.log("🚀 V64 模型训练启动 (EfficientNet-B0 多任务版)")
    oof_status = "跳过OOF筛选" if config['OOF_p_true_threshold'] == 0 else f"OOF_p_true_threshold={config['OOF_p_true_threshold']}"
    logger.log(f"🎯 核心目标: {oof_status}, image_index_threshold={config['image_index_threshold']}")
    logger.log(f"🔬 设备: {device}")
    logger.log(f"📁 模型保存目录: {output_dir}")
    logger.log(f"🖼️ 训练图像根目录: {config['image_root']}")
    if 'verify_root' in config:
        logger.log(f"🧪 验证图像根目录: {config['verify_root']}")
    else:
        logger.log(f"🧪 验证图像根目录: {config['image_root']} (与训练相同)")

    # --- 数据加载 ---
    logger.log(f"\n📊 Loading datasets:")

    # 加载训练数据集
    sop4_data_path = config['sop4_data']
    logger.log(f"  - Loading training data from: {sop4_data_path}")
    df_train_raw = read_csv_with_encoding(sop4_data_path, logger)
    logger.log(f"    - Found {len(df_train_raw)} raw training records.")

    # 应用所有筛选规则
    df_train_filtered = apply_all_filters(df_train_raw, config, df_name="Training Data")

    # 加载验证数据集
    verify_data_path = config['verify_data']
    logger.log(f"  - Loading verification data from: {verify_data_path}")
    df_verify_raw = read_csv_with_encoding(verify_data_path, logger)
    logger.log(f"    - Found {len(df_verify_raw)} raw verification records.")

    # 应用所有筛选规则 (跳过时间过滤)
    df_verify_filtered = apply_all_filters(df_verify_raw, config, df_name="Verification Data", skip_time_filter=True)

    # 数据统计
    logger.log(f"\n📈 最終訓練数据统计:")
    logger.log(f"  总样本数: {len(df_train_filtered)}")

    # 数据源统计 (如果存在)
    if 'dataset_source' in df_train_filtered.columns:
        source_counts = df_train_filtered['dataset_source'].value_counts()
        logger.log(f"  数据源分布:")
        for source, count in source_counts.items():
            logger.log(f"    {source}: {count} ({count/len(df_train_filtered)*100:.1f}%)")

    # BOM统计
    if 'bom' in df_train_filtered.columns:
        bom_counts = df_train_filtered['bom'].value_counts().sort_index()
        bom_valid = df_train_filtered['bom'].notna().sum()
        logger.log(f"  BOM有效样本: {bom_valid} ({bom_valid/len(df_train_filtered)*100:.1f}%)")
        for bom_val, count in bom_counts.items():
            bom_name = "良性" if bom_val == 0 else "恶性"
            logger.log(f"    {bom_name}(BOM={bom_val}): {count} ({count/bom_valid*100:.1f}%)")

    # 统计其他特征
    feature_mapping = json.load(open(config['feature_mapping_file'], 'r', encoding='utf-8'))
    for task in ['ti_rads', 'composition', 'echo', 'foci', 'margin', 'shape']:
        if task in df_train_filtered.columns:
            valid_count = df_train_filtered[task].notna().sum()
            logger.log(f"  {task.capitalize()}有效样本: {valid_count} ({valid_count/len(df_train_filtered)*100:.1f}%)")

    # 数据划分 - 按access_no分组，避免数据泄露
    logger.log(f"\n🛡️ 按access_no分组划分，避免数据泄露...")

    # 统计access_no分布
    access_groups = df_train_filtered['access_no'].nunique()
    logger.log(f"  数据集包含 {access_groups} 个不同的access_no")

    # 使用GroupShuffleSplit按access_no分组
    gss = GroupShuffleSplit(n_splits=1, test_size=config['test_size'], random_state=config['random_state'])
    train_idx, val_idx = next(gss.split(df_train_filtered, groups=df_train_filtered['access_no']))

    train_df = df_train_filtered.iloc[train_idx].reset_index(drop=True)
    val_df = df_train_filtered.iloc[val_idx].reset_index(drop=True)

    # 验证分组效果
    train_groups = train_df['access_no'].nunique()
    val_groups = val_df['access_no'].nunique()
    logger.log(f"  训练集access_no: {train_groups} 个")
    logger.log(f"  验证集access_no: {val_groups} 个")

    # 检查是否有重叠
    overlap = set(train_df['access_no'].unique()) & set(val_df['access_no'].unique())
    if len(overlap) == 0:
        logger.log(f"  ✅ 数据划分成功，无access_no重叠")
    else:
        logger.log(f"  ⚠️ 警告：发现 {len(overlap)} 个重叠的access_no")

    # 创建数据集
    train_transform, val_transform = get_transforms_v60()
    
    # 检查是否使用LMDB加速
    use_lmdb = config.get('use_lmdb', False)
    train_lmdb_exists = os.path.exists(config.get('train_lmdb_path', ''))
    
    if use_lmdb and train_lmdb_exists:
        logger.log(f"\n⚡ 使用LMDB高速数据加载模式")
        logger.log(f"   LMDB路径: {config['train_lmdb_path']}")
        # LMDB已包含过滤后的数据，直接使用划分索引
        train_set = NoduleFeatureDataset_LMDB(config['train_lmdb_path'], feature_mapping, train_transform, indices=train_idx)
        val_set = NoduleFeatureDataset_LMDB(config['train_lmdb_path'], feature_mapping, val_transform, indices=val_idx)
    else:
        if use_lmdb:
            logger.log(f"\n⚠️ LMDB文件不存在，使用传统文件加载模式")
            logger.log(f"   请运行: python create_lmdb_dataset.py --filter 创建LMDB数据集")
        else:
            logger.log(f"\n📁 使用传统文件加载模式")
        train_set = NoduleFeatureDataset_V60(train_df, config['image_root'], feature_mapping, train_transform)
        val_set = NoduleFeatureDataset_V60(val_df, config['image_root'], feature_mapping, val_transform)

    # DataLoader配置 - 针对CUDA多GPU环境优化
    # LMDB模式下可以使用更多worker，因为I/O不再是瓶颈
    num_workers = config['num_workers'] if not (use_lmdb and train_lmdb_exists) else min(config['num_workers'] * 2, 16)
    loader_kwargs = {
        'batch_size': config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': True if torch.cuda.is_available() else False,
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': 4 if num_workers > 0 else None,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    logger.log(f"\n🔄 最终数据划分:")
    logger.log(f"  训练集: {len(train_set)} 样本 ({train_groups} 个access_no)")
    logger.log(f"  验证集: {len(val_set)} 样本 ({val_groups} 个access_no)")

    # 统计分组后的BOM分布
    train_bom_dist = train_df['bom'].value_counts().sort_index()
    val_bom_dist = val_df['bom'].value_counts().sort_index()
    logger.log(f"  训练集BOM分布: 良性{train_bom_dist.get(0, 0)}, 恶性{train_bom_dist.get(1, 0)}")
    logger.log(f"  验证集BOM分布: 良性{val_bom_dist.get(0, 0)}, 恶性{val_bom_dist.get(1, 0)}")

    # 准备定期验证
    logger.log(f"\n🧪 Setting up periodic verification on independent test set...")
    _, verify_transform = get_transforms_v60()

    # 确定验证图像根目录
    verify_image_root = config.get('verify_root', config['image_root'])
    verify_lmdb_exists = os.path.exists(config.get('verify_lmdb_path', ''))
    
    if use_lmdb and verify_lmdb_exists:
        logger.log(f"  - Using LMDB for verification: {config['verify_lmdb_path']}")
        verify_dataset = NoduleFeatureDataset_LMDB(config['verify_lmdb_path'], feature_mapping, verify_transform)
    else:
        logger.log(f"  - Using file-based verification: {verify_image_root}")
        verify_dataset = NoduleFeatureDataset_V60(df_verify_filtered, verify_image_root, feature_mapping, verify_transform)
    
    verify_loader = DataLoader(verify_dataset, shuffle=False, **loader_kwargs)
    logger.log(f"  - Verification will run every {config['verify_epoch_interval']} epochs on {len(verify_dataset)} samples.")

    # 模型初始化
    model = MultiTaskNoduleCNN_V60(feature_mapping, dropout_rate=config['dropout_rate']).to(device)
    
    # 多GPU并行训练支持
    num_gpus = get_num_gpus()
    if config['use_multi_gpu'] and num_gpus > 1 and torch.cuda.is_available():
        gpu_ids = config['gpu_ids'] if config['gpu_ids'] else list(range(num_gpus))
        logger.log(f"\n🚀 启用多GPU并行训练: 使用 {len(gpu_ids)} 个GPU")
        logger.log(f"    GPU IDs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
        # 调整有效batch_size
        effective_batch_size = config['batch_size'] * len(gpu_ids)
        logger.log(f"    有效批次大小: {config['batch_size']} x {len(gpu_ids)} = {effective_batch_size}")
    else:
        logger.log(f"\n🔧 单GPU/CPU训练模式")
    
    loss_fn = FocusedLossManager_V60(config['loss_weights'], device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=7, min_lr=1e-6)

    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"\n🏗️ 模型架构 (EfficientNet-B0 多任务版):")
    logger.log(f"  总参数: {total_params/1e6:.2f}M")
    logger.log(f"  可训练参数: {trainable_params/1e6:.2f}M")
    logger.log(f"  参数/样本比: {total_params/len(train_set):.0f}:1")

    logger.log(f"\n⚖️ 损失权重配置:")
    for task, weight in config['loss_weights'].items():
        logger.log(f"  - {task.capitalize():<12}: {weight}")

    # --- 训练循环 ---
    logger.log(f"\n🎯 开始训练 (早停轮数: {config['early_stop_patience']}):")
    num_epochs = config['num_epochs']
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = config['early_stop_patience']

    for epoch in range(num_epochs):
        # 训练和验证
        train_loss, train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, config)
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, device)

        current_auc = val_metrics.get('bom_auc', 0)
        scheduler.step(current_auc)

        # 打印结果
        print_results(epoch, train_metrics, val_metrics)

        # 可视化样本
        visualize_epoch_samples(model, val_loader, device, epoch, output_dir, set_name="val")

        # TensorBoard记录
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Train/BOM_AUC', train_metrics.get('bom_auc', 0), epoch)
        writer.add_scalar('Val/BOM_AUC', val_metrics.get('bom_auc', 0), epoch)
        writer.add_scalar('Train/BOM_Accuracy', train_metrics.get('bom_accuracy', 0), epoch)
        writer.add_scalar('Val/BOM_Accuracy', val_metrics.get('bom_accuracy', 0), epoch)
        writer.add_scalar('Train/BOM_Sensitivity', train_metrics.get('bom_sensitivity', 0), epoch)
        writer.add_scalar('Val/BOM_Sensitivity', val_metrics.get('bom_sensitivity', 0), epoch)
        writer.add_scalar('Train/BOM_Specificity', train_metrics.get('bom_specificity', 0), epoch)
        writer.add_scalar('Val/BOM_Specificity', val_metrics.get('bom_specificity', 0), epoch)

        # 记录辅助任务指标 (处理DataParallel包装)
        model_mappings = model.module.mappings if isinstance(model, nn.DataParallel) else model.mappings
        for task in model_mappings.keys():
            if task == 'bom': continue
            train_acc = train_metrics.get(f'{task}_accuracy', 0)
            val_acc = val_metrics.get(f'{task}_accuracy', 0)
            writer.add_scalar(f'Train/{task.capitalize()}_Accuracy', train_acc, epoch)
            writer.add_scalar(f'Val/{task.capitalize()}_Accuracy', val_acc, epoch)

        # 定期在独立验证集上进行验证
        if (epoch + 1) % config['verify_epoch_interval'] == 0:
            logger.log(f"\n--- 🧪 Periodic Verification on Independent Test Set (Epoch {epoch + 1}) ---")
            _, periodic_verify_metrics = validate_epoch(model, verify_loader, loss_fn, device)

            logger.log(f"  - AUC: {periodic_verify_metrics.get('bom_auc', 0):.4f}, "
                  f"Acc: {periodic_verify_metrics.get('bom_accuracy', 0):.4f}, "
                  f"Sens: {periodic_verify_metrics.get('bom_sensitivity', 0):.4f}, "
                  f"Spec: {periodic_verify_metrics.get('bom_specificity', 0):.4f}")

            # 记录到TensorBoard
            writer.add_scalar('Verify/BOM_AUC', periodic_verify_metrics.get('bom_auc', 0), epoch)
            writer.add_scalar('Verify/BOM_Accuracy', periodic_verify_metrics.get('bom_accuracy', 0), epoch)
            writer.add_scalar('Verify/BOM_Sensitivity', periodic_verify_metrics.get('bom_sensitivity', 0), epoch)
            writer.add_scalar('Verify/BOM_Specificity', periodic_verify_metrics.get('bom_specificity', 0), epoch)

            # 记录辅助任务的独立验证集准确率 (处理DataParallel包装)
            model_mappings = model.module.mappings if isinstance(model, nn.DataParallel) else model.mappings
            for task in model_mappings.keys():
                if task == 'bom': continue
                verify_acc = periodic_verify_metrics.get(f'{task}_accuracy', 0)
                writer.add_scalar(f'Verify/{task.capitalize()}_Accuracy', verify_acc, epoch)

        # 模型保存和早停
        if current_auc > best_auc:
            best_auc = current_auc
            best_epoch = epoch + 1
            patience_counter = 0
            # 记录最佳epoch信息
            logger.log_best_epoch(epoch, train_metrics, val_metrics)
            # 保存模型时处理DataParallel包装
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, f'nodule_feature_cnn_{config["model_name"]}_best_auc.pth'))
            logger.log(f"  ✅ 新的最佳模型已保存 (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            logger.log(f"🛑 早停触发 ({early_stop_patience}轮无改善)!")
            break

    writer.close()
    logger.log(f"\n🎉 V64 训练完成!")
    logger.log(f"🏆 最佳 BOM AUC: {best_auc:.4f}")

    # --- 最终验证 ---
    logger.log(f"\n\n--- 最终模型验证 ---")
    logger.log(f"🚀 对独立验证集进行最终性能评估: {config['verify_data']}")

    # 加载最佳模型
    best_model_path = os.path.join(output_dir, f'nodule_feature_cnn_{config["model_name"]}_best_auc.pth')
    if os.path.exists(best_model_path):
        logger.log(f"  - Loading best model from: {best_model_path}")
        # 在加载state_dict前，需要先实例化一个同样结构的模型
        model_for_eval = MultiTaskNoduleCNN_V60(feature_mapping, dropout_rate=config['dropout_rate']).to(device)
        model_for_eval.load_state_dict(torch.load(best_model_path, map_location=device))

        logger.log(f"  - Verifying on {len(verify_dataset)} samples...")

        # 验证
        _, verify_metrics = validate_epoch(model_for_eval, verify_loader, loss_fn, device)

        # 记录最终验证结果
        logger.log_final_verify(verify_metrics)

        logger.log("\n\n--- 最终验证性能 ---")
        logger.log(f"  - 数据集: {config['verify_data']}")
        logger.log(f"  - 样本数: {len(verify_dataset)}")
        logger.log("  --------------------")
        logger.log(f"  - BOM AUC:         {verify_metrics.get('bom_auc', 0):.4f}")
        logger.log(f"  - BOM Accuracy:    {verify_metrics.get('bom_accuracy', 0):.4f}")
        logger.log(f"  - BOM Sensitivity: {verify_metrics.get('bom_sensitivity', 0):.4f}")
        logger.log(f"  - BOM Specificity: {verify_metrics.get('bom_specificity', 0):.4f}")

        # 打印其他任务的准确率 (处理DataParallel包装)
        final_model_mappings = model_for_eval.module.mappings if isinstance(model_for_eval, nn.DataParallel) else model_for_eval.mappings
        for task in sorted(final_model_mappings.keys()):
            if task == 'bom': continue
            acc = verify_metrics.get(f'{task}_accuracy', 0)
            logger.log(f"  - {task.capitalize():<12} Acc: {acc:.4f}")

        logger.log("  --------------------")
    else:
        logger.log(f"  - 🔴 Error: Best model not found at {best_model_path}")

    logger.log(f"📊 V64训练特色:")
    # 获取实际模型参数（处理DataParallel包装）
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    logger.log(f"  - EfficientNet-B0多任务架构 ({sum(p.numel() for p in actual_model.parameters())/1e6:.2f}M参数)")
    logger.log(f"  - 7个分类任务 (BOM, TI-RADS, 5个征象)")
    oof_desc = "跳过OOF筛选" if config['OOF_p_true_threshold'] == 0 else f"OOF_p_true_threshold={config['OOF_p_true_threshold']}"
    logger.log(f"  - {oof_desc}, image_index_threshold={config['image_index_threshold']}")
    logger.log(f"  - 复杂的五重数据筛选策略")
    logger.log(f"  - 嵌入特征映射便于推理")
    logger.log(f"📁 模型路径: {best_model_path}")

    # 保存训练总结报告
    logger.save_training_summary(config, best_auc, best_epoch)

if __name__ == "__main__":
    main()
