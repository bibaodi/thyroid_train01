#!/usr/bin/env python3
"""
SOP5 OOF Analysis Script
K-fold Cross-Validation with Probability Calibration for Suspect Sample Detection

This script performs:
1. K-fold cross-validation using EfficientNet-B0
2. Probability calibration using temperature scaling
3. Suspect sample detection (p_true < 0.1 and predicted_class != bom)
4. Saves results with all computed fields for traceability
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import warnings
from tqdm import tqdm
import logging
from datetime import datetime

# Add project root to path
project_root = "/Users/mouxiaoyong/Documents/PycharmProjects/us_feature_classify_v2"
sys.path.append(project_root)

# Import improved logging
from improved_logger import ImprovedLogger

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# 运行模式配置: 'local' 或 'remote'
RUN_MODE = 'remote'  # 修改这里来切换模式

# 不同模式的配置
MODE_CONFIGS = {
    'local': {
        'sop5_dataset': '/datasets/sop-all-single-nodule_v3_batch_detect_v1_nodule_couple_wo_20240809.xlsx',
        'nodule_image_root': '/Users/mouxiaoyong/Documents/images/all-single-nodule/nodule_images',
        'output_file': 'data/dataset_sop5/sop4_all_single_nodule_with_OOF_suspect.csv',
    },
    'remote': {
        'sop5_dataset': '/data/dataInTrain/251016-efficientNet/dataset_table/train/all_matched_sops_ds_v3_with_tr13_1016.csv',
        'nodule_image_root': '/data/dataInTrain/251016-efficientNet/dataset_images/us_images',
        'output_file': '/tmp/output-all_matched_sops_ds_v3_with_tr13_0926_with_OOF_suspect.csv',
    }
}

# 基础配置
CONFIG = {
    # 运行模式
    'run_mode': RUN_MODE,

    # Model parameters
    'model_name': 'EfficientNet-B0',
    'num_classes': 2,  # Binary classification (benign/malignant)
    'input_size': 224,
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,

    # K-fold parameters
    'n_folds': 5,
    'random_state': 42,

    # Probability calibration
    'calibration_method': 'temperature_scaling',
    'suspect_threshold': 0.1,

    # Data validation
    'validate_images': True,  # Set to False to skip image validation step

    # Device and performance
    'device': 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    'num_workers': 0 if torch.backends.mps.is_available() else 4,  # MPS doesn't support multiprocessing
    'pin_memory': False if torch.backends.mps.is_available() else True,  # MPS doesn't support pin_memory

    # Logging
    'log_level': 'INFO',
    'log_dir': 'data/dataset_sop6/logs_0922',
    'log_name': 'sop5_OOF_analysis_v4'
}

# 合并模式特定的配置
CONFIG.update(MODE_CONFIGS[RUN_MODE])

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging():
    """Setup improved logging configuration"""
    improved_logger = ImprovedLogger(
        name=CONFIG['log_name'],
        log_dir=CONFIG['log_dir'],
        log_level=CONFIG['log_level']
    )
    return improved_logger.setup()

logger = setup_logging()

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config():
    """验证配置的有效性"""
    logger.info("=" * 60)
    logger.info("CONFIGURATION VALIDATION")
    logger.info("=" * 60)

    # 显示当前运行模式
    logger.info(f"🔧 运行模式: {CONFIG['run_mode'].upper()}")
    logger.info(f"📁 数据集文件: {CONFIG['sop5_dataset']}")
    logger.info(f"🖼️ 图像根目录: {CONFIG['nodule_image_root']}")
    logger.info(f"💾 输出文件: {CONFIG['output_file']}")

    # 检查数据集文件是否存在
    if not os.path.exists(CONFIG['sop5_dataset']):
        logger.error(f"❌ 数据集文件不存在: {CONFIG['sop5_dataset']}")
        raise FileNotFoundError(f"数据集文件不存在: {CONFIG['sop5_dataset']}")
    else:
        logger.info(f"✅ 数据集文件存在")

    # 检查图像根目录是否存在
    if not os.path.exists(CONFIG['nodule_image_root']):
        logger.error(f"❌ 图像根目录不存在: {CONFIG['nodule_image_root']}")
        if CONFIG['validate_images']:
            raise FileNotFoundError(f"图像根目录不存在: {CONFIG['nodule_image_root']}")
        else:
            logger.warning("⚠️ 图像验证已禁用，将跳过图像检查")
    else:
        logger.info(f"✅ 图像根目录存在")

    # 创建输出目录
    output_dir = os.path.dirname(CONFIG['output_file'])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"✅ 输出目录已准备: {output_dir}")

    logger.info("=" * 60)

# =============================================================================
# DEVICE DETECTION AND SETUP
# =============================================================================
def setup_device():
    """Setup and log device information"""
    device = CONFIG['device']
    
    logger.info("=" * 50)
    logger.info("DEVICE CONFIGURATION")
    logger.info("=" * 50)
    
    if device == 'cuda':
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif device == 'mps':
        logger.info("Using Apple Metal Performance Shaders (MPS)")
        logger.info("MPS is optimized for Apple Silicon Macs")
    else:
        logger.info("Using CPU")
        logger.info(f"CPU Cores: {torch.get_num_threads()}")
    
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info("=" * 50)
    
    return device

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_dataset(file_path):
    """智能加载数据集，自动识别CSV和Excel格式"""
    logger.info(f"Loading dataset from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            # 尝试不同编码读取CSV
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"✅ CSV文件读取成功 (编码: {encoding}): {len(df)} 条记录")
                    break
                except Exception as e:
                    logger.debug(f"编码 {encoding} 失败: {e}")
                    continue

            if df is None:
                raise ValueError(f"无法使用任何编码读取CSV文件: {file_path}")

        elif file_ext in ['.xlsx', '.xls']:
            # 读取Excel文件
            try:
                df = pd.read_excel(file_path)
                logger.info(f"✅ Excel文件读取成功: {len(df)} 条记录")
            except Exception as e:
                logger.error(f"读取Excel文件失败: {e}")
                raise
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}. 支持的格式: .csv, .xlsx, .xls")

        # 基本验证
        if len(df) == 0:
            raise ValueError("数据集为空")

        logger.info(f"数据集基本信息:")
        logger.info(f"  - 行数: {len(df)}")
        logger.info(f"  - 列数: {len(df.columns)}")
        logger.info(f"  - 内存使用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        return df

    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

# =============================================================================
# DATA VALIDATION AND FILTERING
# =============================================================================
def validate_and_filter_dataset(dataframe, image_root):
    """Validate dataset by checking which images actually exist and filter out missing ones"""
    logger.info("Validating dataset - checking for existing images...")
    logger.info(f"Image root directory: {image_root}")
    logger.info(f"DataFrame shape before validation: {dataframe.shape}")
    logger.info(f"DataFrame index range: {dataframe.index.min()} to {dataframe.index.max()}")

    # Check if image root exists
    if not os.path.exists(image_root):
        logger.error(f"Image root directory does not exist: {image_root}")
        raise FileNotFoundError(f"Image root directory does not exist: {image_root}")

    valid_rows = []
    missing_images = []
    processed_count = 0

    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Checking images"):
        try:
            access_no = str(row['access_no'])
            sop_uid = str(row['sop_uid'])
            image_path = os.path.join(image_root, access_no, f"{sop_uid}.jpg")

            if os.path.exists(image_path):
                valid_rows.append(row.to_dict())  # Convert to dict to avoid index issues
            else:
                missing_images.append({
                    'index': idx,
                    'access_no': access_no,
                    'sop_uid': sop_uid,
                    'image_path': image_path
                })

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            logger.error(f"Row data: {row.to_dict()}")
            continue

    logger.info(f"Processed {processed_count} rows")
    logger.info(f"Found {len(valid_rows)} valid images")
    logger.info(f"Found {len(missing_images)} missing images")

    # Create new dataframe from valid rows
    if valid_rows:
        filtered_dataframe = pd.DataFrame(valid_rows).reset_index(drop=True)
    else:
        logger.warning("No valid images found, returning empty DataFrame")
        filtered_dataframe = pd.DataFrame()
    
    logger.info(f"Image validation complete:")
    logger.info(f"  Original samples: {len(dataframe)}")
    logger.info(f"  Valid samples: {len(filtered_dataframe)}")
    logger.info(f"  Missing images: {len(missing_images)}")
    
    if missing_images:
        logger.warning(f"Found {len(missing_images)} samples with missing images")
        # Save list of missing images for reference
        missing_file = CONFIG['output_file'].replace('.csv', '_missing_images.txt')
        with open(missing_file, 'w') as f:
            f.write("Missing Images Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total missing: {len(missing_images)}\n\n")
            for item in missing_images[:10]:  # Show first 10
                f.write(f"Index {item['index']}: {item['image_path']}\n")
            if len(missing_images) > 10:
                f.write(f"... and {len(missing_images) - 10} more\n")
        logger.info(f"Missing images list saved to: {missing_file}")
    
    return filtered_dataframe

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
class NoduleDataset(Dataset):
    """Custom dataset for nodule images with masks"""
    
    def __init__(self, dataframe, image_root, transform=None, mask_transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load image
        access_no = str(row['access_no'])
        sop_uid = str(row['sop_uid'])
        image_path = os.path.join(self.image_root, access_no, f"{sop_uid}.jpg")
        
        # Load mask
        mask_path = os.path.join(self.image_root, access_no, f"{sop_uid}_mask.jpg")
        
        # Load RGB image (already validated to exist)
        image = Image.open(image_path).convert('RGB')
        
        # Load mask (grayscale)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # Create dummy mask if not found
            mask = Image.new('L', image.size, 255)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Get labels
        bom_label = int(row['bom'])  # Primary label (0: benign, 1: malignant)
        ti_rads_label = int(row['ti_rads']) if pd.notna(row['ti_rads']) else 1  # Secondary label (1-5)
        
        return {
            'image': image,
            'mask': mask,
            'bom_label': bom_label,
            'ti_rads_label': ti_rads_label,
            'access_no': access_no,
            'sop_uid': sop_uid,
            'index': idx
        }

def get_transforms():
    """Get image transforms for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
        transforms.ToTensor()
    ])
    
    return train_transform, val_transform, mask_transform

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 based classifier"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = efficientnet_b0(pretrained=pretrained)
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# =============================================================================
# TRAINING AND VALIDATION FUNCTIONS
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        labels = batch['bom_label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        probability_logi, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate model for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            labels = batch['bom_label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_labels, all_probabilities

# =============================================================================
# PROBABILITY CALIBRATION
# =============================================================================
class TemperatureScaling:
    """Temperature scaling for probability calibration"""
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def fit(self, logits, labels):
        """Fit temperature scaling on validation set"""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        return self.temperature.item()
    
    def predict_proba(self, logits):
        """Get calibrated probabilities"""
        return torch.softmax(logits / self.temperature, dim=1)

# =============================================================================
# K-FOLD CROSS VALIDATION
# =============================================================================
def perform_kfold_cv(dataframe, image_root):
    """Perform K-fold cross-validation with probability calibration"""
    
    logger.info(f"Starting {CONFIG['n_folds']}-fold cross-validation")
    logger.info(f"Total samples: {len(dataframe)}")
    
    # Initialize results storage
    all_results = []
    
    # Get transforms
    train_transform, val_transform, mask_transform = get_transforms()
    
    # Stratified K-fold split
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['random_state'])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataframe, dataframe['bom'])):
        logger.info(f"Processing Fold {fold + 1}/{CONFIG['n_folds']}")
        
        # Split data
        train_df = dataframe.iloc[train_idx].reset_index(drop=True)
        val_df = dataframe.iloc[val_idx].reset_index(drop=True)
        
        logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        # Create datasets
        train_dataset = NoduleDataset(train_df, image_root, train_transform, mask_transform)
        val_dataset = NoduleDataset(val_df, image_root, val_transform, mask_transform)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
        
        # Initialize model
        model = EfficientNetClassifier(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(CONFIG['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
            
            # Validate
            val_loss, val_acc, val_preds, val_labels, val_probs = validate_epoch(model, val_loader, criterion, CONFIG['device'])
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Get final predictions on validation set
        model.eval()
        val_predictions = []
        val_probabilities = []
        val_logits = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Final prediction Fold {fold + 1}"):
                images = batch['image'].to(CONFIG['device'])
                outputs = model(images)
                
                logits = outputs
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                val_logits.extend(logits.cpu().numpy())
                val_probabilities.extend(probabilities.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
        
        # Apply temperature scaling for calibration
        logger.info("Applying temperature scaling calibration")
        temp_scaler = TemperatureScaling()
        val_logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
        
        temperature = temp_scaler.fit(val_logits_tensor, val_labels_tensor)
        calibrated_probs = temp_scaler.predict_proba(val_logits_tensor)
        
        logger.info(f"Temperature scaling factor: {temperature:.4f}")
        
        # Store results for this fold
        for i, (idx, row) in enumerate(val_df.iterrows()):
            original_idx = val_idx[i]
            p_true = calibrated_probs[i][row['bom']].item()  # Probability of true class
            predicted_class = val_predictions[i]
            
            # Determine if suspect
            suspect = 1 if (p_true < CONFIG['suspect_threshold'] and predicted_class != row['bom']) else 0
            
            result = {
                'original_index': original_idx,
                'access_no': row['access_no'],
                'sop_uid': row['sop_uid'],
                'bom': row['bom'],
                'ti_rads': row['ti_rads'],
                'fold': fold + 1,
                'predicted_class': predicted_class,
                'p_true': p_true,
                'p_benign': calibrated_probs[i][0].item(),
                'p_malignant': calibrated_probs[i][1].item(),
                'temperature': temperature,
                'suspect': suspect,
                'correct_prediction': 1 if predicted_class == row['bom'] else 0
            }
            
            all_results.append(result)
    
    return all_results

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function"""
    logger.info("Starting SOP5 OOF Analysis")

    # Validate configuration first
    validate_config()

    # Setup device and log information
    device = setup_device()
    
    try:
        # Load dataset using smart loader
        dataframe = load_dataset(CONFIG['sop5_dataset'])
        
        # Check required columns
        required_columns = ['access_no', 'sop_uid', 'bom', 'ti_rads']
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Filter out rows with missing essential data
        initial_count = len(dataframe)
        dataframe = dataframe.dropna(subset=['access_no', 'sop_uid', 'bom'])
        logger.info(f"Filtered dataset (missing data): {initial_count} -> {len(dataframe)} samples")
        
        # Validate and filter dataset by checking existing images (if enabled)
        if CONFIG['validate_images']:
            dataframe = validate_and_filter_dataset(dataframe, CONFIG['nodule_image_root'])

            # Check if we have any data left after validation
            if len(dataframe) == 0:
                logger.error("No valid images found after validation. Please check image paths.")
                raise ValueError("No valid images found after validation")
        else:
            logger.info("Image validation skipped (validate_images=False)")

        # Check data distribution after filtering
        if len(dataframe) > 0:
            bom_distribution = dataframe['bom'].value_counts()
            logger.info(f"BOM distribution (after image validation): {bom_distribution.to_dict()}")

            # Check if we have both classes
            if len(bom_distribution) < 2:
                logger.warning(f"Only one class found in BOM distribution: {bom_distribution.to_dict()}")
        else:
            logger.error("No data remaining after filtering")
            raise ValueError("No data remaining after filtering")
        
        # Perform K-fold cross-validation
        results = perform_kfold_cv(dataframe, CONFIG['nodule_image_root'])
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original data to preserve all columns
        final_df = dataframe.merge(results_df, left_index=True, right_on='original_index', how='left')
        
        # Add analysis summary
        total_samples = len(final_df)
        suspect_samples = final_df['suspect'].sum()
        correct_predictions = final_df['correct_prediction'].sum()
        overall_accuracy = correct_predictions / total_samples
        
        logger.info(f"Analysis Summary:")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Suspect samples: {suspect_samples} ({suspect_samples/total_samples*100:.2f}%)")
        logger.info(f"Correct predictions: {correct_predictions} ({overall_accuracy*100:.2f}%)")
        
        # Save results
        logger.info(f"Saving results to {CONFIG['output_file']}")
        final_df.to_csv(CONFIG['output_file'], index=False)
        
        # Save summary statistics
        summary_file = CONFIG['output_file'].replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("SOP5 OOF Analysis Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Dataset: {CONFIG['sop5_dataset']}\n")
            f.write(f"Model: {CONFIG['model_name']}\n")
            f.write(f"K-fold CV: {CONFIG['n_folds']} folds\n")
            f.write(f"Calibration: {CONFIG['calibration_method']}\n")
            f.write(f"Suspect threshold: {CONFIG['suspect_threshold']}\n")
            f.write(f"Image validation: {'Enabled' if CONFIG['validate_images'] else 'Disabled'}\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Suspect samples: {suspect_samples} ({suspect_samples/total_samples*100:.2f}%)\n")
            f.write(f"Overall accuracy: {overall_accuracy*100:.2f}%\n")
            f.write(f"Output file: {CONFIG['output_file']}\n")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
