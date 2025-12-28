#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import inspect
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, matthews_corrcoef
from torch.utils.tensorboard import SummaryWriter  
import warnings
import matplotlib.pyplot as plt
import random
import re
from datetime import datetime
import io
import math
import gc

# Import YOLO dataset utilities
from yolo_dataset_utils import save_images_in_yolo_format

def debugprint(message):
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    if caller_frame:
        filename = caller_frame.f_code.co_filename
        line_number = caller_frame.f_lineno
        print(f"[{filename}:{line_number}] {message}")
    # Don't forget to clean up
    del frame, caller_frame


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def monitor_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

warnings.filterwarnings('ignore')
def getdatestr():
	from datetime import datetime
	# Get current date
	current_date = datetime.now()
	# Format as '251107'
	formatted_date = current_date.strftime('%y%m%d')
	#print(formatted_date)
	return formatted_date

# =============================================================================
# 配置管理 - V64 配置项统一管理
# =============================================================================
base_model_path=r'/train/src/task4efficientNet/srcs/models/thyUSCls-NoduIMGv251115F1.1/nodule_feature_cnn_v5Nodu_f10.8828.pth'
CONFIG = {
    # 数据集配置
    'sop4_data2': '/data/dataInTrain/251016-efficientNet/dataset_table/train/all_matched_sops_ds_v3_with_tr13_1016.csv',
    'sop4_data': '/data/dataInTrain/251016-efficientNet/dataset_table/all_matched_sop_v5.csv',
    'train_additional_data1':'/train/data/251016-efficientNet/dataset_table/none_single_tr13.csv', #append tirads1-3 as benign data --eton@251114;
    'verify_data': '/data/dataInTrain/251016-efficientNet/dataset_table/val/all_verify_sop_with_predictions.csv',
    'image_root': '/data/dataInTrain/251016-efficientNet/dataset_images/nodule_images',
    'image_root0': '/data/dataInTrain/251016-efficientNet/dataset_images/us_images/',
    '00verify_root': '/Users/Shared/tars/nodule_images/',
    'feature_mapping_file': '/train/src/task4efficientNet/dataset/all_features_mapping_numer_v4.json',

    # 模型配置
    'model_name': 'v5NoduAntiOverfit',
    'output_dir': f"/tmp/thyUSCls-NoduIMGv{getdatestr()}F1.2AntiOverfit",
    'tensorboard_dir': 'runs/nodule_feature_cnn',
    'dropout_rate': 0.5,
    'input_image_size': (224, 224),

    # 训练配置
    'batch_size': 256,
    'num_epochs': 30,
    'learning_rate': 1e-3,
    'weight_decay': 2e-2,
    'early_stop_patience': 10,
    'max_grad_norm': 1.0,
    'verify_epoch_interval': 3,

    # 数据筛选配置
    'OOF_p_true_threshold': 0.2,
    'image_index_threshold': 16,
    'exclude_months': ['202408', '202409'],
    'exclude_ti_rads': [6],
    'minDropImageEdge':16,

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
    'num_workers': 2,  # 避免多进程问题

    # Anti-overfitting configuration
    'freeze_backbone_layers': True,  # Whether to freeze backbone layers
    'num_layers_to_freeze': 5,  # Number of layers to freeze
    'use_cosine_scheduler': True,  # Use cosine annealing learning rate scheduler
    'warmup_epochs': 3,  # Number of warmup epochs
    'cross_validation': False,  # Whether to use cross-validation
    'n_splits': 5  # Number of cross-validation folds
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

    def save_training_summary(self, config, best_f1, total_epochs):
        """保存训练总结到文件"""
        try:
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Anti-Overfitting Optimization 甲状腺结节特征CNN模型训练报告\n")
                f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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
                f.write(f"  最佳验证MCC: {best_f1:.4f}\n")
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
                    f.write(f"    BOM F1: {train_metrics.get('bom_f1', 0):.4f}\n")
                    f.write(f"    BOM Accuracy: {train_metrics.get('bom_accuracy', 0):.4f}\n")
                    f.write(f"    BOM Sensitivity: {train_metrics.get('bom_sensitivity', 0):.4f}\n")
                    f.write(f"    BOM Specificity: {train_metrics.get('bom_specificity', 0):.4f}\n")

                    f.write("  验证集指标:\n")
                    f.write(f"    BOM F1: {val_metrics.get('bom_f1', 0):.4f}\n")
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
    """获取最佳可用设备"""
    cudev = torch.device("cpu")
    if torch.backends.mps.is_available(): 
        cudev = torch.device("mps")
    if torch.cuda.is_available(): 
        cudev = torch.device("cuda")
    print(f"torch-device={cudev}")
    return cudev 

def load_config(cfg=None):
    """加载配置"""
    if cfg:
        return cfg
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

def read_csv_with_encoding(file_path, frac=1.0, logger=None):
    """
    尝试使用不同编码读取CSV文件
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if logger:
                logger.log(f"    - Successfully loaded with {encoding} encoding. percent={frac}")
            if 0:
                # 选择前50%的行
                half_length = int(len(df) * 0.5)
                df = df.iloc[:half_length]
            # 随机选择50%的行（设置random_state以确保结果可重现）
            df = df.sample(frac=frac, random_state=34)
            
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Failed to read {file_path} with any of the tried encodings: {encodings}")

def concat_csv_with_column_alignment(base_df, new_csv_path, logger=None):
    """
    Read a new CSV file and concatenate it to base_df with proper column alignment.
    
    Args:
        base_df: Base DataFrame to concatenate to
        new_csv_path: Path to the new CSV file to read
        logger: Optional logger for logging messages
        
    Returns:
        Concatenated DataFrame with aligned columns
        
    Rules:
        - Only keeps columns that exist in base_df
        - Adds missing columns with NaN values
        - Drops extra columns from new_df that don't exist in base_df
    """
    if logger:
        logger.log(f"\n📎 Concatenating additional data from: {new_csv_path}")
    else:
        print(f"\n📎 Concatenating additional data from: {new_csv_path}")
    
    # Read the new CSV file
    try:
        new_df = read_csv_with_encoding(new_csv_path, 1.0, logger)
    except Exception as e:
        error_msg = f"  ❌ Failed to read {new_csv_path}: {e}"
        if logger:
            logger.log(error_msg)
        else:
            print(error_msg)
        return base_df
    
    if logger:
        logger.log(f"  - Loaded {len(new_df)} records from new file")
    else:
        print(f"  - Loaded {len(new_df)} records from new file")
    
    # Get column sets
    base_columns = set(base_df.columns)
    new_columns = set(new_df.columns)
    
    # Find column differences
    common_columns = base_columns & new_columns
    missing_in_new = base_columns - new_columns
    extra_in_new = new_columns - base_columns
    
    # Log column analysis
    if logger:
        logger.log(f"  - Column analysis:")
        logger.log(f"    • Common columns: {len(common_columns)}")
        if missing_in_new:
            logger.log(f"    • Missing in new file (will add with NaN): {len(missing_in_new)}")
            if len(missing_in_new) <= 10:
                logger.log(f"      {sorted(missing_in_new)}")
        if extra_in_new:
            logger.log(f"    • Extra in new file (will drop): {len(extra_in_new)}")
            if len(extra_in_new) <= 10:
                logger.log(f"      {sorted(extra_in_new)}")
    else:
        print(f"  - Common columns: {len(common_columns)}")
        if missing_in_new:
            print(f"  - Missing in new file: {len(missing_in_new)}")
        if extra_in_new:
            print(f"  - Extra in new file (will drop): {len(extra_in_new)}")
    
    # Add missing columns to new_df with NaN
    for col in missing_in_new:
        new_df[col] = np.nan
    
    # Keep only columns that exist in base_df (drop extra columns)
    new_df_aligned = new_df[list(base_df.columns)]
    
    # Concatenate
    result_df = pd.concat([base_df, new_df_aligned], ignore_index=True)
    
    if logger:
        logger.log(f"  ✅ Concatenation complete: {len(base_df)} + {len(new_df_aligned)} = {len(result_df)} records")
    else:
        print(f"  ✅ Concatenation complete: {len(base_df)} + {len(new_df_aligned)} = {len(result_df)} records")
    
    return result_df


def concat_multiple_csv_files(base_csv_path, additional_csv_paths, logger=None, frac=1.0):
    """
    Read and concatenate multiple CSV files with column alignment.
    
    Args:
        base_csv_path: Path to the base CSV file
        additional_csv_paths: List of paths to additional CSV files
        logger: Optional logger for logging messages
        
    Returns:
        Concatenated DataFrame with aligned columns
    """
    if logger:
        logger.log(f"\n📚 Loading and concatenating multiple CSV files...")
    
    # Load base file
    df_result = read_csv_with_encoding(base_csv_path, frac,  logger)
    if logger:
        logger.log(f"  - Base file: {len(df_result)} records from {base_csv_path}")
    
    # Concatenate additional files
    for csv_path in additional_csv_paths:
        if os.path.exists(csv_path):
            df_result = concat_csv_with_column_alignment(df_result, csv_path, logger)
        else:
            warning_msg = f"  ⚠️  File not found, skipping: {csv_path}"
            if logger:
                logger.log(warning_msg)
            else:
                print(warning_msg)
    
    return df_result


def validate_and_filter_images(df, image_root, df_name="DataFrame", min_edge_size:int = 16):
    """
    Validate dataset by checking which images actually exist and filter out missing ones.
    
    Args:
        df: DataFrame with 'access_no' and 'sop_uid' columns
        image_root: Root directory for images
        df_name: Name of the dataset for logging
        
    Returns:
        Filtered DataFrame with only existing images
    """
    print(f"\n🔍 Validating images for {df_name}...")
    print(f"  - Image root: {image_root}")
    print(f"  - Original samples: {len(df)};  Minimum required image edge size: {min_edge_size}" )
    
    if not os.path.exists(image_root):
        print(f"  ⚠️  WARNING: Image root directory does not exist: {image_root}")
        print(f"  - Skipping image validation")
        return df
    
    valid_indices = []
    missing_count = 0
    small_size_count = 0
    
    for idx, row in df.iterrows():
        try:
            access_no = str(row['access_no'])
            sop_uid = str(row['sop_uid'])
            image_path = os.path.join(image_root, access_no, f"{sop_uid}.jpg")
            
            if os.path.exists(image_path):
                if  isinstance(min_edge_size, int) and min_edge_size > 0:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        if width < min_edge_size or height < min_edge_size:
                            small_size_count += 1
                            continue
                
                valid_indices.append(idx)
            else:
                missing_count += 1
        except Exception as e:
            print(f"  ⚠️  Error processing row {idx}: {e}")
            missing_count += 1
            continue
    
    # Filter dataframe to keep only valid images
    df_filtered = df.loc[valid_indices].reset_index(drop=True)
    
    print(f"  - Missing images: {missing_count}, Images below size threshold: {small_size_count}")
    print(f"  - Valid samples: {len(df_filtered)}, Retention rate: {len(df_filtered)/len(df)*100:.2f}%")
    
    return df_filtered


def apply_all_filters(df, config, df_name="DataFrame", skip_time_filter=False):
    """
    对给定的DataFrame应用所有筛选规则.
    """
    print(f"\nApplying all filters to {df_name}...")
    original_count = len(df)
    if original_count<1:
        return df
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

class KeepAspectRatioResize:
    def __init__(self, target_width=224, target_height=224):
        self.target_width = target_width
        self.target_height = target_height
    
    def __call__(self, image):
        w, h = image.size
        if w <= 0 or h <= 0:
            print(f"Warning: Invalid image size ({w}x{h}), creating blank image")
            return Image.new('RGB', (self.target_width, self.target_height), (0, 0, 0))
        
        scale = min(self.target_width / w, self.target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w <= 0 or new_h <= 0:
            print(f"Warning: Calculated size ({new_w}x{new_h}) invalid, using target size")
            new_w = self.target_width
            new_h = self.target_height
        
        image = transforms.Resize((new_h, new_w))(image)
        
        pad_w = (self.target_width - new_w) // 2
        pad_h = (self.target_height - new_h) // 2
        
        pad_transform = transforms.Pad(
            (pad_w, pad_h, self.target_width - new_w - pad_w, self.target_height - new_h - pad_h), 
            fill=0
        )
        return pad_transform(image)

def get_transforms_v60():
    """V60 数据增强策略"""
    target_image_size = CONFIG['input_image_size']
    target_W = target_image_size[0]
    target_H = target_image_size[1]

    def resize_and_pad(image, target_width=224, target_height=224):
        # function cannot be pickled, so we define a class instead
        w, h = image.size
        
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = transforms.Resize((new_h, new_w))(image)
        pad_w = (target_width - new_w) // 2
        pad_h = (target_height - new_h) // 2
        
        pad_transform = transforms.Pad(
            (pad_w, pad_h, target_width - new_w - pad_w, target_height - new_h - pad_h), 
            fill=0
        )
        return pad_transform(image)
    
    resize_and_pad_224x224 = KeepAspectRatioResize(target_width=target_W, target_height=target_H)

    train_transform = transforms.Compose([
        resize_and_pad_224x224, #transforms.Lambda(resize_and_pad_pack),
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])
    
    val_transform = transforms.Compose([
        resize_and_pad_224x224, #transforms.Lambda(resize_and_pad_pack),
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
        
        # 添加文件存在检查
        self._filter_existing_images()
        
        self.feature_mapping = feature_mapping
        self.transform = transform
        
        self.tasks = list(feature_mapping.keys())
        
        # 为每个征象创建反向映射，用于将文本标签转换为整数
        self.label_to_int_maps = self._create_label_maps()
    
    def _filter_existing_images(self):
        """检查图像文件是否存在，并过滤掉不存在的行"""
        # 创建一个掩码，标记哪些行的图像文件存在
        image_exists_mask = []
        missing_count = 0
        
        for _, row in self.df.iterrows():
            image_path = os.path.join(self.image_root, str(row['access_no']), f"{row['sop_uid']}.jpg")
            exists = os.path.isfile(image_path)
            image_exists_mask.append(exists)
            
            if not exists:
                missing_count += 1
        
        # 创建新的DataFrame，只包含图像存在的行
        original_len = len(self.df)
        self.df = self.df[image_exists_mask].reset_index(drop=True)
        filtered_count = original_len - len(self.df)
        
        print(f"过滤前样本数: {original_len}, 过滤后样本数: {len(self.df)}, 缺失图像数: {filtered_count}")

    def _create_label_maps(self):
        label_maps = {}
        # 'bom' 和 'ti_rads' 已经在数值映射文件中定义好了
        """
        label_maps: {
            'bom': {'0': 0, '1': 1}, 
            'ti_rads': {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}, 
            'composition': {'囊性': 0, '囊实性': 1, '实性': 2}, 
            'echo': {'海绵样': 0, '无回声': 1, '高回声': 2, '等回声': 3, '强回声': 4, '不均质回声': 5, '低回声': 6, '极低回声': 7}, 
            'foci': {'无点状强回声': 0, '彗星尾征': 1, '无回声区': 2, '高回声区': 3, '粗大钙化': 4, '斑状强回声': 5, '弧形强回声/环状强回声': 6, '点状强回声': 7}, 
            'margin': {'清楚': 0, '尚清': 1, '欠清': 2, '不清': 3}, 'shape': {'规则': 0, '尚规则': 1, '欠规则': 2, '不规则': 3}
        }
        """
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
            if image.size[0] <= 0 or image.size[1] <= 0:
                print(f"Warning: Invalid image dimensions at {image_path}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0)) # Return a black image
            print(f"image not found:{image_path}")

        if self.transform:
            image = self.transform(image)

        item = {'image': image, 'access_no': row['access_no'], 'sop_uid': row['sop_uid']}

        # 为每个任务获取标签和有效性标志
        #bom 1; ti_rads -1; composition -1; echo -1; foci -1; margin -1; shape -1;;
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

# --- V60 EfficientNet-B0 多任务模型 ---
class MultiTaskNoduleCNN_V60(nn.Module):
    """V60 EfficientNet-B0模型 - 支持7个分类任务"""
    def __init__(self, feature_mappings, dropout_rate=0.5):
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
            nn.Dropout(dropout_rate * 0.8)
        )

        # 动态创建分类头
        self.heads = nn.ModuleDict()
        for task, mapping in self.mappings.items():
            num_classes = len(mapping)
            self.heads[task] = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_features(features)

        outputs = {}
        for task, head in self.heads.items():
            outputs[task] = head(shared)

        return outputs

def freeze_backbone_layers(model, num_layers=5):
    """Freeze the first few layers of the backbone network to reduce overfitting"""
    # Freeze the first num_layers blocks of efficientnet
    count = 0
    for name, param in model.backbone.named_parameters():
        # Freeze the first few blocks in the features section
        if 'features' in name:
            # Simply freeze based on parameter name or index
            # Here we approximately freeze the first few layers based on parameter count
            if count < num_layers * 20:  # Each layer has approximately 20 parameter groups
                param.requires_grad = False
            count += 1
    return model

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing learning rate scheduler with warmup"""
    def __init__(self, optimizer, T_max, warmup_epochs, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Warmup phase: linear increase from 0 -> base_lr
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]

        # After warmup: cosine annealing to eta_min over the remaining epochs
        # Protect against T_max <= warmup_epochs
        remaining = max(1, self.T_max - self.warmup_epochs)
        t = self.last_epoch - self.warmup_epochs
        # Clip t to [0, remaining]
        if t < 0:
            t = 0
        if t > remaining:
            t = remaining

        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / remaining)) / 2
            for base_lr in self.base_lrs
        ]
        

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

suppressDebug=0
# --- 训练和验证函数 ---
def train_epoch(model, loader, loss_fn, optimizer, device, config=None):
    model.train()
    total_loss = 0
    metrics_calc = DetailedMetricsCalculator_V60(model.mappings)
    metrics_calc.reset()  # 确保重置

    for batch in tqdm(loader, desc="Training", leave=False):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        global suppressDebug
        if 'image' in batch and 0 == suppressDebug:
            print(f"Batch device: {batch['image'].device}")
            print(f"Batch is on CUDA: {batch['image'].is_cuda}")
            suppressDebug +=1

        optimizer.zero_grad()
        outputs = model(batch['image'])
        losses = loss_fn(outputs['bom'], batch['bom'])
        if isinstance(losses, dict) and 'total' in losses:
            losses['total'].backward()
        else:
            losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'] if config else 1.0)
        optimizer.step()
        if isinstance(losses, dict) and 'total' in losses:
            total_loss += losses['total'].item()
        else:
            total_loss = losses

        metrics_calc.update(outputs, batch)

    return total_loss / len(loader), metrics_calc.compute_metrics()

def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    metrics_calc = DetailedMetricsCalculator_V60(model.mappings)
    metrics_calc.reset()  # 确保重置

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch['image'])
            #losses = loss_fn(outputs, batch)
            losses = loss_fn(outputs['bom'], batch['bom'])
            if isinstance(losses, dict) and 'total' in losses:
                total_loss += losses['total'].item()
            else:
                total_loss = losses
                
            metrics_calc.update(outputs, batch)

    return total_loss / len(loader), metrics_calc.compute_metrics()

# --- V60指标计算器 ---
class DetailedMetricsCalculator_V60:
    def __init__(self, mappings):
        self.mappings = mappings
        self.tasks = list(self.mappings.keys())
        self.reset()

    def reset(self):
        # Use flattened Python lists for robust accumulation across batches
        self.targets = {task: [] for task in self.tasks}  # list of ints
        self.preds = {task: [] for task in self.tasks}    # list of ints
        # bom needs probabilities for AUC calculation
        self.bom_probs = []  # list of floats (probability of positive class)

    def update(self, outputs, batch):
        # Robust per-batch handling: convert to CPU numpy lists and extend
        for task in self.tasks:
            valid_mask = None
            valid_flag_key = f'{task}_valid'
            if valid_flag_key in batch:
                valid_mask = (batch[valid_flag_key] > 0)
            else:
                # If no valid flag, assume all valid
                valid_mask = torch.ones(len(batch['image']), dtype=torch.bool, device=batch['image'].device)

            if valid_mask.sum().item() == 0:
                continue

            # labels
            labels_t = batch[task][valid_mask].detach().cpu().numpy().tolist()
            # preds
            preds_t = torch.argmax(outputs[task], dim=1)[valid_mask].detach().cpu().numpy().tolist()

            self.targets[task].extend(labels_t)
            self.preds[task].extend(preds_t)

            if task == 'bom':
                # positive-class probability (class index 1)
                probs = torch.softmax(outputs['bom'], dim=1)[:, 1][valid_mask].detach().cpu().numpy().tolist()
                self.bom_probs.extend(probs)


    def compute_metrics(self):
        metrics = {}

        for task in self.tasks:
            tgt_list = self.targets[task]
            pred_list = self.preds[task]

            if len(tgt_list) == 0:
                continue

            targets = np.asarray(tgt_list)
            preds = np.asarray(pred_list)

            # Ensure same length
            min_len = min(len(targets), len(preds))
            if min_len == 0:
                continue
            targets = targets[:min_len]
            preds = preds[:min_len]

            if task == 'bom':
                # Compute AUC if possible
                metrics['bom_auc'] = float('nan')
                if len(self.bom_probs) >= min_len:
                    try:
                        probs = np.asarray(self.bom_probs)[:min_len]
                        if len(np.unique(targets)) > 1:
                            metrics['bom_auc'] = float(roc_auc_score(targets, probs))
                    except Exception:
                        metrics['bom_auc'] = float('nan')

                # F1 (binary) - set NaN if only one class present to highlight instability
                if len(np.unique(targets)) < 2:
                    metrics['bom_f1'] = float('nan')
                    metrics['bom_mcc'] = float('nan')
                else:
                    try:
                        metrics['bom_f1'] = float(f1_score(targets, preds, average='binary', pos_label=1, zero_division=0))
                    except Exception as e:
                        print(f"Error computing bom_f1: {e}")
                        metrics['bom_f1'] = float('nan')
                    try:
                        metrics['bom_mcc'] = float(matthews_corrcoef(targets, preds))
                    except Exception as e:
                        print(f"Error computing bom_mcc: {e}")
                        metrics['bom_mcc'] = float('nan')

                # Confusion matrix / sensitivity / specificity / accuracy
                try:
                    if len(np.unique(targets)) > 1:
                        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
                        metrics['bom_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        metrics['bom_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        metrics['bom_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
                    else:
                        # Only one class present
                        metrics['bom_sensitivity'] = float('nan')
                        metrics['bom_specificity'] = float('nan')
                        metrics['bom_accuracy'] = float(preds[0] == targets[0])
                        metrics['bom_mcc'] = float('nan')
                except Exception as e:
                    print(f"Error computing confusion metrics for BOM: {e}")
                    metrics['bom_sensitivity'] = float('nan')
                    metrics['bom_specificity'] = float('nan')
                    metrics['bom_accuracy'] = float('nan')
            else:
                # For auxiliary tasks, compute accuracy (handles multi-class)
                try:
                    metrics[f'{task}_accuracy'] = float((preds == targets).mean())
                except Exception as e:
                    print(f"Error computing accuracy for {task}: {e}")
                    metrics[f'{task}_accuracy'] = float('nan')

        return metrics

def print_results(epoch, train_metrics, val_metrics):
    print(f"\n📊 Epoch {epoch+1} Results (V64):")
    # BOM 指标 - 处理NaN值
    def format_metric(value, default=0):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
        return f"{value:.4f}"

    train_auc = format_metric(train_metrics.get('bom_f1'))
    train_mcc = format_metric(train_metrics.get('bom_mcc'))
    train_acc = format_metric(train_metrics.get('bom_accuracy'))
    train_sens = format_metric(train_metrics.get('bom_sensitivity'))
    train_spec = format_metric(train_metrics.get('bom_specificity'))

    val_auc = format_metric(val_metrics.get('bom_f1'))
    val_mcc = format_metric(val_metrics.get('bom_mcc'))
    val_acc = format_metric(val_metrics.get('bom_accuracy'))
    val_sens = format_metric(val_metrics.get('bom_sensitivity'))
    val_spec = format_metric(val_metrics.get('bom_specificity'))

    print(f"  🎯 Training:   MCC={train_mcc}, Acc={train_acc}, F1={train_auc}, Sens={train_sens}, Spec={train_spec}")
    print(f"  🔍 Validation: MCC={val_mcc}, Acc={val_acc}, F1={val_auc}, Sens={val_sens}, Spec={val_spec}")

    # 辅助任务指标
    aux_tasks = [k.replace('_accuracy', '') for k in val_metrics.keys() if '_accuracy' in k and 'bom' not in k]
    for task in sorted(aux_tasks):
        train_acc = train_metrics.get(f'{task}_accuracy', 0)
        val_acc = val_metrics.get(f'{task}_accuracy', 0)
        print(f"    - {task.capitalize():<12} Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

# --- 可视化函数 (参考V37) ---
def create_image_grid(images, labels, allChPredictions, access_nos, sop_uids, title, save_path):
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
            y_bom_pred = allChPredictions[idx] #[benign, malignant]
            pred_malignant_val = y_bom_pred[1]
            #print("y-bom-pred:", y_bom_pred, pred_malignant_val)
            if 0:
                pred_class = 1 if pred_malignant_val > 0.5 else 0
            else:
                pred_class = torch.argmax(y_bom_pred).item()
            color = 'green' if bom_label == pred_class else 'red'

            # 左上角显示BOM标签
            ax.text(0.05, 0.95, f"BOM: {bom_label}", transform=ax.transAxes,
                   color='white', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7),
                   va='top')

            # 右上角显示预测概率
            ax.text(0.95, 0.95, f"{pred_malignant_val:.3f}", transform=ax.transAxes,
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
            probs = torch.softmax(outputs['bom'], dim=1) #[:, 1] keep [benign-logit, malignant-logit] format

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
                test_probsidx=probs[idx]
                bom_pred_ret=test_probsidx.cpu()
                #print(" _3item=", bom_pred_ret.item() , "RuntimeError: a Tensor with 2 elements cannot be converted to Scalar")
                #print("PROBS[idx]=", type(test_probsidx), type(bom_pred_ret),bom_pred_ret  )
                predictions.append(bom_pred_ret)
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

def count_stats(df_series, typeName="BOM", logger=None):
    bom_counts = df_series.value_counts().sort_index()
    bom_valid = df_series.notna().sum()
    logger.log(f"  {typeName}有效样本: {bom_valid} ({bom_valid/len(df_series)*100:.1f}%)")
    for bom_val, count in bom_counts.items():
        bom_name = "良性" if bom_val == 0 else "恶性"
        logger.log(f"    {bom_name}({typeName}={bom_val}): {count} ({count/bom_valid*100:.1f}%)")

def initialize_misc(config):
    """Initialize training environment, device, output directory, and logger."""
    device = get_device()
    print(f"acceculate device={device} ")
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(config['tensorboard_dir'])

    # Initialize logger
    log_file_path = os.path.join(output_dir, f'training_report_{config["model_name"]}.txt')
    logger = TrainingLogger(log_file_path)

    train_dataImgRoot = config['image_root']
    logger.log("🚀 V64 模型训练启动 (EfficientNet-B0 多任务版)")
    oof_status = "跳过OOF筛选" if config['OOF_p_true_threshold'] == 0 else f"OOF_p_true_threshold={config['OOF_p_true_threshold']}"
    logger.log(f"🎯 核心目标: {oof_status}, image_index_threshold={config['image_index_threshold']}")
    logger.log(f"🔬 设备: {device}")
    logger.log(f"📁 模型保存目录: {output_dir}")
    logger.log(f"🖼️ 训练图像根目录: {train_dataImgRoot}")
    if 'verify_root' in config:
        logger.log(f"🧪 验证图像根目录: {config['verify_root']}")
    else:
        logger.log(f"🧪 验证图像根目录: {train_dataImgRoot} (与训练相同)")
            
    logger.log(f"\n⚖️ model-损失权重配置:")
    for task, weight in config['loss_weights'].items():
        logger.log(f"  - {task.capitalize():<12}: {weight}")

    
    return device, output_dir, writer, logger, train_dataImgRoot

def static_dataset_info(df_train_filtered, logger=None):
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
        count_stats(df_train_filtered['bom'], "BOM", logger)
    # 统计其他特征    
    for task in ['ti_rads', 'composition', 'echo', 'foci', 'margin', 'shape']:
        if task in df_train_filtered.columns:
            valid_count = df_train_filtered[task].notna().sum()
            logger.log(f"  {task.capitalize()}有效样本: {valid_count} ({valid_count/len(df_train_filtered)*100:.1f}%)")
            
def load_and_prepare_data(config, train_dataImgRoot, logger):
    """Load, filter, and prepare training and verification datasets."""
    logger.log(f"\n📊 Loading datasets:")

    # Load training dataset
    sop4_data_path = config['sop4_data']
    additional_paths = [
        config.get('train_additional_data1'),
        config.get('train_additional_data2'),
    ]
    additional_paths = [p for p in additional_paths if p]

    if additional_paths:
        df_train_raw = concat_multiple_csv_files(sop4_data_path, additional_paths, logger, 0.02)
    else:
        logger.log(f"  - Loading training data from: {sop4_data_path}")
        df_train_raw = read_csv_with_encoding(sop4_data_path, 1.0,  logger)
        logger.log(f"    - Found {len(df_train_raw)} raw training records.")

    min_edge_size = 16
    if config and 'minDropImageEdge' in config:
        min_edge_size = config['minDropImageEdge']
        print(f"  - Minimum image edge size: {min_edge_size}")
    # Validate image existence and filter
    df_train_filtered = validate_and_filter_images(df_train_raw, train_dataImgRoot, df_name="Training Data", min_edge_size=min_edge_size)
    
    if 0: # Save images in YOLO classification format organized by BOM (with train/val split)
        yolo_output_dir = r'/opt/dlami/nvme/yolo_format_dataset'
        save_images_in_yolo_format(
            df_train_filtered, 
            train_dataImgRoot, 
            yolo_output_dir, 
            class_column='bom', 
            df_name="Training Data",
            split_name='train',
            val_split=0.2  # 80% train, 20% val - creates both train/ and val/ subdirectories
        )
    
    # Apply all filtering rules
    df_train_filtered = apply_all_filters(df_train_filtered, config, df_name="Training Data")
    static_dataset_info(df_train_filtered, logger)
    
    return df_train_filtered 

def split_dataset_by_access_no(df_train_filtered, config, logger):
    """Split dataset into training and validation sets based on access_no to prevent data leakage.
    
    Args:
        df_train_filtered: Filtered training dataframe
        config: Configuration dictionary containing test_size and random_state
        logger: Logger instance for logging information
    
    Returns:
        tuple: (train_df, val_df) - Training and validation dataframes
    """
    logger.log(f"\n🛡️ Splitting by access_no to prevent data leakage...")
    # Count unique access_no
    access_groups = df_train_filtered['access_no'].nunique()
    logger.log(f"  Dataset contains {access_groups} unique access_no")

    # Use GroupShuffleSplit to split by access_no groups
    gss = GroupShuffleSplit(n_splits=1, test_size=config['test_size'], random_state=config['random_state'])
    train_idx, val_idx = next(gss.split(df_train_filtered, groups=df_train_filtered['access_no']))

    train_df = df_train_filtered.iloc[train_idx].reset_index(drop=True)
    val_df = df_train_filtered.iloc[val_idx].reset_index(drop=True)
    
    # Log BOM statistics if available
    if 'bom' in df_train_filtered.columns:
        count_stats(train_df['bom'], "BOM-In-Train", logger)
        count_stats(val_df['bom'], "BOM-In-Val", logger)

    # Validate train/val split
    train_groups = train_df['access_no'].nunique()
    val_groups = val_df['access_no'].nunique()
    logger.log(f"  Training set access_no: {train_groups}")
    logger.log(f"  Validation set access_no: {val_groups}")

    # Check for overlaps
    overlap = set(train_df['access_no'].unique()) & set(val_df['access_no'].unique())
    if len(overlap) == 0:
        logger.log(f"  ✅ Data split successful, no access_no overlap")
    else:
        logger.log(f"  ⚠️ Warning: Found {len(overlap)} overlapping access_no")
    
    return train_df, val_df

def create_datasets_and_loaders(config, train_df, val_df, train_dataImgRoot, feature_mapping, logger):
    """Create datasets and data loaders for training and validation."""
    # Create datasets
    train_transform, val_transform = get_transforms_v60()
    train_set = NoduleFeatureDataset_V60(train_df, train_dataImgRoot, feature_mapping, train_transform)
    val_set = NoduleFeatureDataset_V60(val_df, train_dataImgRoot, feature_mapping, val_transform)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=2,  # Changed from 0
            pin_memory=False,  # Changed from True
            persistent_workers=True,
            prefetch_factor=2)

    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=2,  # Changed from 0
            pin_memory=False,  # Changed from True
            persistent_workers=True,
            prefetch_factor=2)

    logger.log(f"\n🔄 最终数据划分:")
    logger.log(f"  训练集: {len(train_set)} 样本")
    logger.log(f"  验证集: {len(val_set)} 样本")

    # 统计分组后的BOM分布
    train_bom_dist = train_df['bom'].value_counts().sort_index()
    val_bom_dist = val_df['bom'].value_counts().sort_index()
    logger.log(f"  训练集BOM分布: 良性{train_bom_dist.get(0, 0)}, 恶性{train_bom_dist.get(1, 0)}")
    logger.log(f"  验证集BOM分布: 良性{val_bom_dist.get(0, 0)}, 恶性{val_bom_dist.get(1, 0)}")
    
    return train_set, val_set, train_loader, val_loader

def setup_model_and_optimizers(config, device, feature_mapping, logger):
    """Initialize model, loss function, optimizer and scheduler."""
    # Model initialization
    model = MultiTaskNoduleCNN_V60(feature_mapping, dropout_rate=config['dropout_rate']).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    print(f"\n\n base model=", base_model_path, f"Model device: {next(model.parameters()).device}",f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    
    # Freeze part of the backbone network layers
    if config['freeze_backbone_layers']:
        model = freeze_backbone_layers(model, config['num_layers_to_freeze'])
        logger.log(f"\n🧊 First {config['num_layers_to_freeze']} layers of backbone network frozen to reduce overfitting")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Use improved learning rate scheduler
    if config['use_cosine_scheduler']:
        scheduler = WarmupCosineAnnealingLR(
            optimizer, 
            T_max=config['num_epochs'], 
            warmup_epochs=config['warmup_epochs'],
            eta_min=1e-6
        )
        logger.log(f"\n📈 Using cosine annealing learning rate scheduler with warmup ({config['warmup_epochs']} epochs)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=7, min_lr=1e-6)

    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"\n🏗️ 模型架构 (EfficientNet-B0 多任务版):")
    logger.log(f"  总参数: {total_params/1e6:.2f}M")
    logger.log(f"  可训练参数: {trainable_params/1e6:.2f}M")
    #logger.log(f"  参数/样本比: {total_params/len(train_set):.0f}:1")
    
    return model, loss_fn, optimizer, scheduler

def log_to_tensorboard(writer, epoch, train_loss, val_loss, train_metrics, val_metrics, model):
    """Log metrics to TensorBoard."""
    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Val/Loss', val_loss, epoch)
    writer.add_scalar('Train/BOM_F1', train_metrics.get('bom_f1', 0), epoch)
    writer.add_scalar('Val/BOM_F1', val_metrics.get('bom_f1', 0), epoch)
    # Log MCC
    writer.add_scalar('Train/BOM_MCC', train_metrics.get('bom_mcc', 0), epoch)
    writer.add_scalar('Val/BOM_MCC', val_metrics.get('bom_mcc', 0), epoch)
    writer.add_scalar('Train/BOM_Accuracy', train_metrics.get('bom_accuracy', 0), epoch)
    writer.add_scalar('Val/BOM_Accuracy', val_metrics.get('bom_accuracy', 0), epoch)
    writer.add_scalar('Train/BOM_Sensitivity', train_metrics.get('bom_sensitivity', 0), epoch)
    writer.add_scalar('Val/BOM_Sensitivity', val_metrics.get('bom_sensitivity', 0), epoch)
    writer.add_scalar('Train/BOM_Specificity', train_metrics.get('bom_specificity', 0), epoch)
    writer.add_scalar('Val/BOM_Specificity', val_metrics.get('bom_specificity', 0), epoch)

    # Log auxiliary task metrics
    for task in model.mappings.keys():
        if task == 'bom': continue
        train_acc = train_metrics.get(f'{task}_accuracy', 0)
        val_acc = val_metrics.get(f'{task}_accuracy', 0)
        writer.add_scalar(f'Train/{task.capitalize()}_Accuracy', train_acc, epoch)
        writer.add_scalar(f'Val/{task.capitalize()}_Accuracy', val_acc, epoch)

def perform_periodic_verification(model, verify_loader, loss_fn, device, epoch, writer, logger):
    """Perform periodic verification on independent test set."""
    logger.log(f"\n--- 🧪 Periodic Verification on Independent Test Set (Epoch {epoch + 1}) ---")
    _, periodic_verify_metrics = validate_epoch(model, verify_loader, loss_fn, device)

    logger.log(f"  - F1: {periodic_verify_metrics.get('bom_f1', 0):.4f}, "
          f"Acc: {periodic_verify_metrics.get('bom_accuracy', 0):.4f}, "
        f"Sens: {periodic_verify_metrics.get('bom_sensitivity', 0):.4f}, "
        f"Spec: {periodic_verify_metrics.get('bom_specificity', 0):.4f}, "
        f"MCC: {periodic_verify_metrics.get('bom_mcc', 0):.4f}")

    # Log to TensorBoard
    writer.add_scalar('Verify/BOM_F1', periodic_verify_metrics.get('bom_f1', 0), epoch)
    writer.add_scalar('Verify/BOM_Accuracy', periodic_verify_metrics.get('bom_accuracy', 0), epoch)
    writer.add_scalar('Verify/BOM_Sensitivity', periodic_verify_metrics.get('bom_sensitivity', 0), epoch)
    writer.add_scalar('Verify/BOM_Specificity', periodic_verify_metrics.get('bom_specificity', 0), epoch)
    writer.add_scalar('Verify/BOM_MCC', periodic_verify_metrics.get('bom_mcc', 0), epoch)

    # Log auxiliary task accuracy on independent verification set
    for task in model.mappings.keys():
        if task == 'bom': continue
        verify_acc = periodic_verify_metrics.get(f'{task}_accuracy', 0)
        writer.add_scalar(f'Verify/{task.capitalize()}_Accuracy', verify_acc, epoch)

def perform_model_training(model, train_loader, val_loader, verify_loader, loss_fn, optimizer, scheduler, 
                config, device, output_dir, writer, logger):
    """Train the model with early stopping and periodic verification."""
    logger.log(f"\n🎯 开始训练 (早停轮数: {config['early_stop_patience']}):")
    num_epochs = config['num_epochs']
    # monitor MCC as primary metric
    best_mcc = -1.0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = config['early_stop_patience']

    for epoch in range(num_epochs):
        # 训练和验证
        train_loss, train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, config)
        val_loss, val_metrics = validate_epoch(model, val_loader, loss_fn, device)

        current_valMCC = val_metrics.get('bom_mcc', 0)
        current_trainMCC = train_metrics.get('bom_mcc', 0)
        # Different step calls based on scheduler type
        if config['use_cosine_scheduler']:
            scheduler.step()
            try:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else scheduler.get_lr()[0]
            except Exception:
                current_lr = 0.0
            logger.log(f"  Current learning rate: {current_lr:.6f}")
        else:
            # ReduceLROnPlateau expects a metric value; pass MCC
            scheduler.step(current_valMCC)

        # 打印结果
        print_results(epoch, train_metrics, val_metrics)

        # 可视化样本
        if False == config.get('use_combined_images', False): 
            visualize_epoch_samples(model, val_loader, device, epoch, output_dir, set_name="val")
        log_to_tensorboard(writer, epoch, train_loss, val_loss, train_metrics, val_metrics, model)

        # 定期在独立验证集上进行验证
        if (epoch + 1) % config['verify_epoch_interval'] == 0:
            perform_periodic_verification(model, verify_loader, loss_fn, device, epoch, writer, logger)
        
        # Save epoch checkpoint (include MCC)
        torch.save(model.state_dict(), os.path.join(output_dir, f'nodule_USfeats_{config["model_name"]}_mccT{current_trainMCC:.3f}-V{current_valMCC:.3f}.pth'))
        # 模型保存和早停 (monitor MCC)
        if current_trainMCC > best_mcc:
            best_mcc = current_trainMCC
            best_epoch = epoch + 1
            patience_counter = 0
            # 记录最佳epoch信息
            logger.log_best_epoch(epoch, train_metrics, val_metrics)
            torch.save(model.state_dict(), os.path.join(output_dir, f'nodule_USfeats_{config["model_name"]}_best_mcc.pth'))
            logger.log(f"  ✅ 新的最佳模型已保存 (tain MCC: {best_mcc:.4f})")
        else:
            logger.log(f" *patience counter:{patience_counter}")
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            logger.log(f"🛑 早停触发 ({early_stop_patience}轮无改善)!")
            break

        clear_gpu_memory()
        monitor_memory()

    writer.close()
    logger.log(f"\n🎉 V64 训练完成!")
    logger.log(f"🏆 最佳 BOM MCC: {best_mcc:.4f}")

    return best_mcc, best_epoch

def load_verification_dataset(config, image_root, min_edge_size=16, logger=None):
    """Load and prepare verification dataset separately.
    
    Args:
        config: Configuration dictionary
        image_root: Root directory for images
        min_edge_size: Minimum required image edge size
        logger: Logger instance
    
    Returns:
        Filtered verification dataframe and verify image root
    """
    verify_data_path = config.get('verify_data')
    if not verify_data_path:
        raise ValueError("verify_data must be specified in config")
        
    if logger:
        logger.log(f"  - Loading verification data from: {verify_data_path}")
    
    df_verify_raw = read_csv_with_encoding(verify_data_path, frac=1.0, logger=logger)
    
    if logger:
        logger.log(f"    - Found {len(df_verify_raw)} raw verification records.")
    
    # Validate image existence and filter
    verify_image_root = config.get('verify_root', image_root)
    df_verify_filtered = validate_and_filter_images(
        df_verify_raw, 
        verify_image_root, 
        df_name="Verification Data", 
        min_edge_size=min_edge_size
    )
    
    # Apply all filtering rules (skip time filtering)
    df_verify_filtered = apply_all_filters(
        df_verify_filtered, 
        config, 
        df_name="Verification Data", 
        skip_time_filter=True
    )
    
    return df_verify_filtered, verify_image_root

def create_independent_test_dataset(config, train_dataImgRoot, feature_mapping, logger):
    """Create verification dataset and loader."""
    logger.log(f"\n🧪 Setting up periodic verification on independent test set...")

    verify_image_root = config.get('verify_root', train_dataImgRoot)
    if 'verify_root' in config:
        logger.log(f"  - Using separate verify_root for verification images: {verify_image_root}")
    else:
        logger.log(f"  - Using same image_root for verification images: {verify_image_root}")    
    # Load verification dataset using the separate function
    df_verify_filtered, verify_image_root = load_verification_dataset(
        config, 
        verify_image_root, 
        min_edge_size=32, 
        logger=logger
    )
    debugprint(f">>>>>>>verify_image_root={verify_image_root}")

    _, verify_transform = get_transforms_v60()
    test_dataset = NoduleFeatureDataset_V60(df_verify_filtered, verify_image_root, feature_mapping, verify_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    logger.log(f"  - Verification will run every {config['verify_epoch_interval']} epochs on {len(test_dataset)} samples.")
    

    return test_dataset, test_loader

def perform_final_verification(config, feature_mapping, device, output_dir, logger, model):
    """Perform final verification on the best model."""
    logger.log(f"\n\n--- 最终模型验证 ---")
    logger.log(f"🚀 对独立验证集进行最终性能评估: {config['verify_data']}")

    # Load best model
    best_model_path = os.path.join(output_dir, f'nodule_feature_cnn_{config["model_name"]}_best_mcc.pth')
    if os.path.exists(best_model_path):
        logger.log(f"  - Loading best model from: {best_model_path}")
        # Instantiate a model with the same structure before loading state_dict
        model_for_eval = MultiTaskNoduleCNN_V60(feature_mapping, dropout_rate=config['dropout_rate']).to(device)
        model_for_eval.load_state_dict(torch.load(best_model_path, map_location=device))

        # Create verification dataset and loader
        verify_dataset ,verify_loader = create_independent_test_dataset(config, config['image_root'], feature_mapping, logger)
        logger.log(f"  - Verifying on {len(verify_dataset)} samples...")
        if len(verify_dataset)<1:
            return
        # Validate
        loss_fn = nn.CrossEntropyLoss()
        _, verify_metrics = validate_epoch(model_for_eval, verify_loader, loss_fn, device)

        # Record final verification results
        logger.log_final_verify(verify_metrics)

        logger.log("\n\n--- 最终验证性能 ---")
        logger.log(f"  - 数据集: {config['verify_data']}")
        logger.log(f"  - 样本数: {len(verify_dataset)}")
        logger.log("  --------------------")
        logger.log(f"  - BOM F1:         {verify_metrics.get('bom_f1', 0):.4f}")
        logger.log(f"  - BOM MCC:        {verify_metrics.get('bom_mcc', 0):.4f}")
        logger.log(f"  - BOM Accuracy:    {verify_metrics.get('bom_accuracy', 0):.4f}")
        logger.log(f"  - BOM Sensitivity: {verify_metrics.get('bom_sensitivity', 0):.4f}")
        logger.log(f"  - BOM Specificity: {verify_metrics.get('bom_specificity', 0):.4f}")

        # Print accuracy of other tasks
        for task in sorted(model.mappings.keys()):
            if task == 'bom': continue
            acc = verify_metrics.get(f'{task}_accuracy', 0)
            logger.log(f"  - {task.capitalize():<12} Acc: {acc:.4f}")

        logger.log("  --------------------")
    else:
        logger.log(f"  - 🔴 Error: Best model not found at {best_model_path}")
    
    return best_model_path

def main():
    """Main training function"""
    config = load_config()
    device, output_dir, writer, logger, train_dataImgRoot = initialize_misc(config)
    df_train_filtered  = load_and_prepare_data(config, train_dataImgRoot, logger)
    train_df, val_df = split_dataset_by_access_no(df_train_filtered, config, logger)
    
    feature_mapping = json.load(open(config['feature_mapping_file'], 'r', encoding='utf-8'))
    train_set, val_set, train_loader, val_loader = create_datasets_and_loaders(
        config, train_df, val_df, train_dataImgRoot, feature_mapping, logger
    )
    test_dataset, test_loader = create_independent_test_dataset(
        config, train_dataImgRoot, feature_mapping, logger
    )

    model, loss_fn, optimizer, scheduler = setup_model_and_optimizers(
        config, device, feature_mapping, logger
    )

    # --- 训练循环 ---
    best_mcc, best_epoch = perform_model_training(
        model, train_loader, val_loader, test_loader, loss_fn, optimizer, scheduler,
        config, device, output_dir, writer, logger
    )

    # --- 最终验证 ---
    best_model_path = perform_final_verification(
        config, feature_mapping, device, output_dir, logger, model
    )

    logger.log(f"📊 V64训练特色:")
    logger.log(f"  - EfficientNet-B0多任务架构 ({sum(p.numel() for p in model.parameters())/1e6:.2f}M参数)")
    logger.log(f"  - 7个分类任务 (BOM, TI-RADS, 5个征象)")
    oof_desc = "跳过OOF筛选" if config['OOF_p_true_threshold'] == 0 else f"OOF_p_true_threshold={config['OOF_p_true_threshold']}"
    logger.log(f"  - {oof_desc}, image_index_threshold={config['image_index_threshold']}")
    logger.log(f"  - 复杂的五重数据筛选策略")
    logger.log(f"  - 嵌入特征映射便于推理")
    logger.log(f"📁 模型路径: {best_model_path}")

    # 保存训练总结报告 (best metric is MCC)
    logger.save_training_summary(config, best_mcc, best_epoch)

if __name__ == "__main__":
    main()
