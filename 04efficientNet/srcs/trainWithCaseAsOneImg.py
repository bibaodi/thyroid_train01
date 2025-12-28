#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import inspect
import sys
import json
import pandas as pd
import numpy as np
import torch
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
import sys
import math
import gc
from train_nodule_feature_cnn_model_v75 import (NoduleFeatureDataset_V60,MultiTaskNoduleCNN_V60,
    get_device, load_config, extract_date_from_access_no, 
    extract_image_index_from_sop_uid, add_image_index_column,
    read_csv_with_encoding, concat_csv_with_column_alignment,
    concat_multiple_csv_files, validate_and_filter_images,
    apply_all_filters, static_dataset_info, split_dataset_by_access_no,create_image_grid,
    setup_model_and_optimizers,print_results,train_epoch,validate_epoch,
    log_to_tensorboard, TrainingLogger, getdatestr,clear_gpu_memory, monitor_memory,KeepAspectRatioResize,get_transforms_v60,visualize_epoch_samples,count_stats,
    perform_final_verification, freeze_backbone_layers, WarmupCosineAnnealingLR,initialize_misc, perform_periodic_verification, perform_model_training,create_independent_test_dataset
)

# Import YOLO dataset utilities
from yolo_dataset_utils import save_images_in_yolo_format


warnings.filterwarnings('ignore')

# =============================================================================
# 配置管理 - V64 配置项统一管理
# =============================================================================
base_model_path=r'/train/src/task4efficientNet/results/thyUSCls-NoduCasev251123MCC/nodule_USfeats_v5NoduCaseUnit_best_mcc.pth'
base_model_path=r'/train/src/task4efficientNet/results/thyUSCls-NoduCasev251125MCC/nodule_USfeats_v5NoduCaseUnit_best_mcc.pth'
CONFIG = {
    # 数据集配置
    'sop4_data2': '/data/dataInTrain/251016-efficientNet/dataset_table/train/all_matched_sops_ds_v3_with_tr13_1016.csv',
    'sop4_data': '/data/dataInTrain/251016-efficientNet/dataset_table/shrinked_all_matched_sop_v6.csv',
    'train_additional_data1':'/train/data/251016-efficientNet/dataset_table/shrinked_none_single_tr13_with_orientation.csv', #append tirads1-3 as benign data --eton@251114;
    'verify_data': '/data/dataInTrain/251016-efficientNet/dataset_table/val/all_verify_sop_with_predictions.csv',
    'image_root': '/data/dataInTrain/251016-efficientNet/dataset_images/nodule_images',
    'image_root0': '/data/dataInTrain/251016-efficientNet/dataset_images/us_images/',
    '00verify_root': '/Users/Shared/tars/nodule_images/',
    'feature_mapping_file': '/train/src/task4efficientNet/dataset/all_features_mapping_numer_v4.json',
    'combined_images_root': '/tmp/combined_case_images',
    'use_combined_images': False,

    # 模型配置
    'model_name': 'v5NoduCaseUnit',
    'output_dir': f"results/thyUSCls-NoduCasev{getdatestr()}MCC",
    'tensorboard_dir': 'runs/tensorboard',
    'dropout_rate': 0.3,
    'input_image_size': (224, 224),

    # 训练配置
    'batch_size': 256,
    'num_epochs': 60,
    'learning_rate': 1e-3,
    'weight_decay': 3e-2,
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


def debugprint(message):
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    if caller_frame:
        filename = caller_frame.f_code.co_filename
        line_number = caller_frame.f_lineno
        print(f"[{filename}:{line_number}] {message}")
    # Don't forget to clean up
    del frame, caller_frame

def combine_case_images(df, original_image_root, combined_images_root, image_prefix="case_", logger=None):
    """Combine Lon and Tra images for each case and save as a single image.
    
    Args:
        df: DataFrame containing access_no and orientation columns
        original_image_root: Root directory of original images
        combined_images_root: Root directory to save combined images
        image_prefix: Prefix for combined image filenames
        logger: Optional logger for logging information
    
    Returns:
        Updated DataFrame with a single row per case containing the combined image path
    """
    # Parameter validation
    required_columns = ['access_no', 'orientation', 'sop_uid']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Create output directory if it doesn't exist
    os.makedirs(combined_images_root, exist_ok=True)
    
    # Group by access_no (case)
    grouped = df.groupby('access_no')
    
    if logger:
        logger.log(f"Combining images for {len(grouped)} cases...")
    
    # Create new dataframe with one row per case
    case_rows = []
    processed_count = 0
    failed_count = 0
    
    for access_no, group in tqdm(grouped, desc="Processing cases"):
        try:
            # Get Lon and Tra images
            lon_image = group[group['orientation'] == 'Lon']
            tra_image = group[group['orientation'] == 'Tra']
            
            if len(lon_image) != 1 or len(tra_image) != 1:
                failed_count += 1
                continue
            
            # Get image paths
            lon_sop_uid = lon_image.iloc[0]['sop_uid']
            tra_sop_uid = tra_image.iloc[0]['sop_uid']
            
            lon_path = os.path.join(original_image_root, str(access_no), f"{lon_sop_uid}.jpg")
            tra_path = os.path.join(original_image_root, str(access_no), f"{tra_sop_uid}.jpg")
            
            # Check if both images exist
            if not (os.path.exists(lon_path) and os.path.exists(tra_path)):
                failed_count += 1
                continue
            
            # Load images
            lon_img = Image.open(lon_path).convert('RGB')
            tra_img = Image.open(tra_path).convert('RGB')
            
            # Create combined image (side by side)
            width = max(lon_img.width, tra_img.width)
            height = lon_img.height + tra_img.height
            combined = Image.new('RGB', (width, height), color=(255, 255, 255))
            
            # Paste images
            combined.paste(lon_img, (0, 0))
            combined.paste(tra_img, (0, lon_img.height))
            
            # Save combined image
            combined_filename = f"{image_prefix}{access_no}.jpg"
            combined_path = os.path.join(combined_images_root, combined_filename)
            combined.save(combined_path)
            
            # Create row with combined image info
            # Use the first row's data and update with combined info
            row_data = group.iloc[0].to_dict()
            row_data['combined_image_path'] = combined_filename
            row_data['sop_uid'] = f"combined_{access_no}"  # New unique ID for combined image
            case_rows.append(row_data)
            
            processed_count += 1
            
        except Exception as e:
            if logger:
                logger.log(f"Error processing case {access_no}: {e}")
            failed_count += 1
    
    if logger:
        logger.log(f"Combined image creation complete: {processed_count} cases successful, {failed_count} failed")
    
    # Create new dataframe
    result_df = pd.DataFrame(case_rows)
    return result_df

class CombinedNoduleFeatureDataset(Dataset):
    """Dataset class for combined case images"""
    def __init__(self, df, image_root, feature_mapping, transform=None):
        self.df = self._filter_single_row_per_case(df.copy())
        self.image_root = image_root
        self.feature_mapping = feature_mapping
        self.transform = transform
        self.tasks = list(feature_mapping.keys())
        self.label_to_int_maps = self._create_label_maps()
    def _filter_single_row_per_case(self, df):
        """Filter dataframe to keep only one row per access_no (case)"""
        # Check if the dataframe already has one row per case
        case_counts = df['access_no'].value_counts()
        if (case_counts > 1).any():
            print(f"Filtering dataframe to keep one row per case. Found {len(case_counts)} unique cases.")
            # Keep only the first row for each case
            df = df.drop_duplicates(subset=['access_no'], keep='first')
            print(f"Filtered dataframe contains {len(df)} rows.")
        return df
        
    def _create_label_maps(self):
        label_maps = {}
        for task in self.tasks:
            label_maps[task] = self.feature_mapping[task]
        return label_maps
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Use the combined image path
        if 'combined_image_path' in row:
            image_filename = row['combined_image_path']
        else:
            # Fallback to using access_no with prefix
            image_filename = f"case_{row['access_no']}.jpg"
        
        image_path = os.path.join(self.image_root, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
            if image.size[0] <= 0 or image.size[1] <= 0:
                print(f"Warning: Invalid image dimensions at {image_path}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            print(f"Image not found: {image_path}")
        
        if self.transform:
            image = self.transform(image)
        
        item = {'image': image, 'access_no': row['access_no']}
        
        # Get labels for each task
        for task in self.tasks:
            raw_val = row.get(task, np.nan)
            label_val = -1
            is_valid = 0.0
            
            if pd.notna(raw_val):
                key = str(raw_val) if not isinstance(raw_val, str) else raw_val
                if isinstance(raw_val, float) and raw_val.is_integer():
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


suppressDebug=0



            
def filter_cases_by_orientation(df, logger=None):
    """Filter dataframe to keep only cases with both 'Lon' and 'Tra' orientations.
    
    Args:
        df: DataFrame containing access_no and orientation columns
        logger: Optional logger for logging information
    
    Returns:
        Filtered DataFrame with only cases containing both orientations
    """
    # Parameter validation
    required_columns = ['access_no', 'orientation']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Log initial information
    initial_cases = df['access_no'].nunique()
    if logger:
        logger.log(f"Filtering cases for required orientations (Lon and Tra)...")
        logger.log(f"  Initial cases: {initial_cases}")
    
    # Get cases that have both 'Lon' and 'Tra' orientations
    cases_with_lon = set(df[df['orientation'] == 'Lon']['access_no'])
    cases_with_tra = set(df[df['orientation'] == 'Tra']['access_no'])
    
    # Keep only cases that have both orientations
    valid_cases = cases_with_lon.intersection(cases_with_tra)
    
    # Filter the dataframe to keep only valid cases
    df_filtered = df[df['access_no'].isin(valid_cases)].copy()
    
    # For each valid case, keep exactly one 'Lon' and one 'Tra' entry
    result_rows = []
    for access_no in valid_cases:
        # Get all rows for this case
        case_rows = df_filtered[df_filtered['access_no'] == access_no]
        
        # Keep one 'Lon' row (select the first one if multiple exist)
        lon_rows = case_rows[case_rows['orientation'] == 'Lon']
        if not lon_rows.empty:
            result_rows.append(lon_rows.iloc[0])
        
        # Keep one 'Tra' row (select the first one if multiple exist)
        tra_rows = case_rows[case_rows['orientation'] == 'Tra']
        if not tra_rows.empty:
            result_rows.append(tra_rows.iloc[0])
    
    # Create the final dataframe
    result_df = pd.DataFrame(result_rows)
    
    # Log results
    final_cases = result_df['access_no'].nunique()
    final_records = len(result_df)
    
    if logger:
        logger.log(f"  Valid cases kept: {final_cases} ({final_cases/initial_cases*100:.1f}%)")
        logger.log(f"  Final records: {final_records}")
        logger.log(f"  Removed cases without both orientations: {initial_cases - final_cases}")
    
    return result_df.reset_index(drop=True)



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
        df_train_raw = concat_multiple_csv_files(sop4_data_path, additional_paths, logger, 1.0)#frac
    else:
        logger.log(f"  - Loading training data from: {sop4_data_path}")
        df_train_raw = read_csv_with_encoding(sop4_data_path, 1.0, logger)
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

    df_train_filtered = filter_cases_by_orientation(df_train_filtered, logger)
    use_combined_images = config.get('use_combined_images', True)
    combined_images_root = train_dataImgRoot 
    if use_combined_images: # Filter cases to keep only those with both 'Lon' and 'Tra' orientations
        combined_images_root = config.get('combined_images_root')
        if not combined_images_root:
            raise ValueError("combined_images_root must be specified in config when using combined images")
        
        logger.log(f"\nCombining images for cases...")
        logger.log(f"  Original image root: {train_dataImgRoot}")
        logger.log(f"  Combined image root: {combined_images_root}")
        
        # Combine images 
        df_train_filtered = combine_case_images(
            df_train_filtered,
            train_dataImgRoot,
            combined_images_root,
            logger=logger
        )
        train_dataImgRoot = combined_images_root
        debugprint(f"train image root={train_dataImgRoot}")
        
    static_dataset_info(df_train_filtered, logger)
    
    return df_train_filtered, combined_images_root



def create_datasets_and_loaders(config, train_df, val_df, train_dataImgRoot, feature_mapping, logger):
    """Create datasets and data loaders for training and validation."""
    # Create datasets
    train_transform, val_transform = get_transforms_v60()
    
    use_combined_images = config.get('use_combined_images', True)
    if use_combined_images:
        train_set = CombinedNoduleFeatureDataset(train_df, train_dataImgRoot, feature_mapping, train_transform)
        val_set = CombinedNoduleFeatureDataset(val_df, train_dataImgRoot, feature_mapping, val_transform)
    else:
        # Use original dataset class
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






def main():
    """Main training function"""
    config = load_config(CONFIG)
    device, output_dir, writer, logger, train_dataImgRoot = initialize_misc(config)
    debugprint(f"train image root={train_dataImgRoot}")

    df_train_filtered, combinedImgHome= load_and_prepare_data(config, train_dataImgRoot, logger)
    feature_mapping = json.load(open(config['feature_mapping_file'], 'r', encoding='utf-8'))
    verify_dataset, verify_loader = create_independent_test_dataset(
        config, train_dataImgRoot, feature_mapping, logger
    )
    train_dataImgRoot = combinedImgHome
    train_df, val_df = split_dataset_by_access_no(df_train_filtered, config, logger)
    
    train_set, val_set, train_loader, val_loader = create_datasets_and_loaders(
        config, train_df, val_df, train_dataImgRoot, feature_mapping, logger
    )

    model, loss_fn, optimizer, scheduler = setup_model_and_optimizers(
        config, device, feature_mapping, logger
    )

    # --- 训练循环 ---
    best_mcc, best_epoch = perform_model_training(
        model, train_loader, val_loader, verify_loader, loss_fn, optimizer, scheduler,
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
