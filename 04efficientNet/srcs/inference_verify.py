#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立验证集推理脚本 - 评估训练好的V75模型
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# =============================================================================
# 配置
# =============================================================================
CONFIG = {
    'model_path': 'models/nodule_feature_cnn_v75/nodule_feature_cnn_v75_best_auc.pth',
    'verify_data': 'data/dataset_sop7/all_verify_sop.csv',
    'verify_root': '/Users/Shared/tars/nodule_images/',  # 使用预处理后的结节图像
    'feature_mapping_file': 'core/utils/all_features_mapping_numer_v4.json',
    'batch_size': 32,
    'num_workers': 0,
    'dropout_rate': 0.4,
    'output_report': 'models/nodule_feature_cnn_v75/verification_report_preprocessed.txt',
    'output_csv': 'data/dataset_sop7/all_verify_sop_with_predictions.csv',  # 带预测结果的CSV
    'bom_threshold': 0.5,
}

# =============================================================================
# 工具函数
# =============================================================================

def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def extract_date_from_access_no(access_no):
    """从access_no中提取年月信息"""
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

# =============================================================================
# 数据集预处理
# =============================================================================

def filter_missing_images(df, image_root):
    """
    过滤掉图片不存在的样本
    
    Args:
        df: 原始DataFrame
        image_root: 图像根目录
        
    Returns:
        过滤后的DataFrame和缺失样本信息
    """
    missing_samples = []
    valid_rows = []
    
    print(f"🔍 检查图像完整性...")
    for idx, row in df.iterrows():
        access_no = row['access_no']
        sop_uid = row['sop_uid']
        image_path = os.path.join(image_root, str(access_no), f"{sop_uid}.jpg")
        
        if os.path.exists(image_path):
            valid_rows.append(row)
        else:
            missing_samples.append({
                'index': idx,
                'access_no': str(access_no),
                'sop_uid': str(sop_uid),
                'path': image_path
            })
    
    # 从有效行重新构建DataFrame，确保索引从0开始
    df_filtered = pd.DataFrame(valid_rows).reset_index(drop=True)
    
    # 报告结果
    total_samples = len(df)
    valid_samples = len(df_filtered)
    missing_count = len(missing_samples)
    
    print(f"  总样本数: {total_samples}")
    print(f"  有效样本: {valid_samples} ({valid_samples/total_samples*100:.2f}%)")
    print(f"  缺失样本: {missing_count} ({missing_count/total_samples*100:.2f}%)")
    
    if missing_count > 0:
        print(f"  ⚠️ 警告: {missing_count} 个样本的图像文件不存在，将被跳过")
        if missing_count <= 10:
            print(f"  缺失样本列表:")
            for sample in missing_samples:
                print(f"    - {sample['access_no']}/{sample['sop_uid']}")
        else:
            print(f"  前10个缺失样本:")
            for sample in missing_samples[:10]:
                print(f"    - {sample['access_no']}/{sample['sop_uid']}")
            print(f"    ... 还有 {missing_count - 10} 个缺失样本")
    
    return df_filtered, missing_samples

# =============================================================================
# 数据集类
# =============================================================================

class NoduleFeatureDataset(Dataset):
    """验证数据集类"""
    def __init__(self, df, image_root, feature_mapping, transform=None):
        self.df = df.copy()
        self.image_root = image_root
        self.feature_mapping = feature_mapping
        self.transform = transform
        self.tasks = list(feature_mapping.keys())
        self.label_to_int_maps = self._create_label_maps()

    def _create_label_maps(self):
        label_maps = {}
        for task in self.tasks:
            label_maps[task] = self.feature_mapping[task]
        return label_maps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, str(row['access_no']), f"{row['sop_uid']}.jpg")

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            # 这不应该发生，因为我们在构造时已经过滤了
            raise FileNotFoundError(
                f"Image not found: {image_path}\n"
                f"  access_no: {row['access_no']}\n"
                f"  sop_uid: {row['sop_uid']}"
            )

        if self.transform:
            image = self.transform(image)

        item = {
            'image': image,
            'access_no': row['access_no'],
            'sop_uid': row['sop_uid'],
            'type': row.get('type', 'unknown'),
            'idx': idx
        }

        # 为每个任务获取标签
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

# =============================================================================
# 模型定义
# =============================================================================

class MultiTaskNoduleCNN(nn.Module):
    """EfficientNet-B0多任务模型"""
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

# =============================================================================
# 推理和指标计算
# =============================================================================

def inference_and_collect(model, loader, device, threshold=0.5):
    """
    推理并收集所有结果
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        threshold: 良恶性判断阈值，恶性概率 >= threshold 则判为恶性（默认0.5）
    """
    model.eval()
    
    all_results = {
        'indices': [],
        'types': [],
        'access_nos': [],
        'sop_uids': [],
        'bom_targets': [],
        'bom_preds': [],
        'bom_probs': []
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ncols=80):
            # 移动数据到设备
            images = batch['image'].to(device)
            
            # 推理
            outputs = model(images)
            bom_logits = outputs['bom']
            bom_probs = torch.softmax(bom_logits, dim=1)[:, 1]  # 恶性概率（类别1）
            
            # 使用阈值判断（而不是argmax）
            # 恶性概率 >= threshold → 判为恶性(1)，否则为良性(0)
            bom_preds = (bom_probs >= threshold).long()
            
            # 收集结果
            valid_mask = batch['bom_valid'] > 0
            if valid_mask.sum() > 0:
                all_results['indices'].extend(batch['idx'][valid_mask].cpu().numpy())
                all_results['types'].extend([batch['type'][i] for i in range(len(batch['type'])) if valid_mask[i]])
                all_results['access_nos'].extend([batch['access_no'][i] for i in range(len(batch['access_no'])) if valid_mask[i]])
                all_results['sop_uids'].extend([batch['sop_uid'][i] for i in range(len(batch['sop_uid'])) if valid_mask[i]])
                all_results['bom_targets'].extend(batch['bom'][valid_mask].cpu().numpy())
                all_results['bom_preds'].extend(bom_preds[valid_mask].cpu().numpy())
                all_results['bom_probs'].extend(bom_probs[valid_mask].cpu().numpy())
    
    return all_results

def calculate_metrics(targets, preds, probs):
    """计算BOM指标"""
    metrics = {}
    
    # 确保有足够的数据
    if len(targets) == 0:
        return {'error': 'No valid samples'}
    
    targets = np.array(targets)
    preds = np.array(preds)
    probs = np.array(probs)
    
    # AUC
    if len(np.unique(targets)) >= 2:
        metrics['auc'] = roc_auc_score(targets, probs)
    else:
        metrics['auc'] = float('nan')
    
    # 混淆矩阵指标
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    metrics['tp'] = int(tp)
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['total'] = len(targets)
    metrics['positive'] = int((targets == 1).sum())
    metrics['negative'] = int((targets == 0).sum())
    
    return metrics

# =============================================================================
# 报告生成
# =============================================================================

def save_predictions_to_csv(df_original, df_filtered, all_results, output_path, missing_samples):
    """
    将推理结果写回到原始DataFrame并保存
    
    Args:
        df_original: 原始DataFrame（包含所有样本）
        df_filtered: 过滤后的DataFrame（只有有效图像的样本）
        all_results: 推理结果字典
        output_path: 输出CSV路径
        missing_samples: 缺失图像的样本列表
    """
    # 创建一个副本
    df_output = df_original.copy()
    
    # 初始化新列（所有样本都设为NaN）
    df_output['bom_pred'] = np.nan
    df_output['bom_confidence'] = np.nan
    df_output['prediction_status'] = 'missing_image'  # 默认状态：缺失图像
    
    # 将推理结果填充到对应的样本中
    # all_results中的数据对应df_filtered中的样本
    for idx in range(len(df_filtered)):
        # 获取过滤后DataFrame中的access_no和sop_uid
        row = df_filtered.iloc[idx]
        access_no = row['access_no']
        sop_uid = row['sop_uid']
        
        # 在原始DataFrame中找到对应的行
        mask = (df_output['access_no'] == access_no) & (df_output['sop_uid'] == sop_uid)
        
        if mask.any():
            # 获取预测结果
            pred = all_results['bom_preds'][idx]  # 0 或 1
            prob = all_results['bom_probs'][idx]  # 恶性的概率
            
            # 填充预测结果
            df_output.loc[mask, 'bom_pred'] = int(pred)
            df_output.loc[mask, 'bom_confidence'] = float(prob)
            df_output.loc[mask, 'prediction_status'] = 'predicted'
    
    # 统计信息
    n_predicted = (df_output['prediction_status'] == 'predicted').sum()
    n_missing = (df_output['prediction_status'] == 'missing_image').sum()
    
    # 保存到CSV
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n📊 预测结果已保存:")
    print(f"  输出文件: {output_path}")
    print(f"  总样本数: {len(df_output)}")
    print(f"  已预测: {n_predicted} ({n_predicted/len(df_output)*100:.2f}%)")
    print(f"  缺失图像: {n_missing} ({n_missing/len(df_output)*100:.2f}%)")
    print(f"\n  新增列说明:")
    print(f"    - bom_pred: 模型预测结果 (0=良性, 1=恶性)")
    print(f"    - bom_confidence: 预测置信度 (恶性概率, 0-1之间)")
    print(f"    - prediction_status: 预测状态 ('predicted'=已预测, 'missing_image'=图像缺失)")

def plot_performance_comparison(all_results, output_path):
    """
    绘制不同数据集的性能对比图
    
    Args:
        all_results: 包含所有预测结果的字典
        output_path: 图片保存路径
    """
    # 计算整体指标
    overall_metrics = calculate_metrics(
        all_results['bom_targets'],
        all_results['bom_preds'],
        all_results['bom_probs']
    )
    
    # 按type分组计算指标
    unique_types = sorted(set(all_results['types']))
    type_metrics = {}
    
    for data_type in unique_types:
        type_mask = np.array([t == data_type for t in all_results['types']])
        type_targets = np.array(all_results['bom_targets'])[type_mask]
        type_preds = np.array(all_results['bom_preds'])[type_mask]
        type_probs = np.array(all_results['bom_probs'])[type_mask]
        
        if len(type_targets) > 0:
            type_metrics[data_type] = calculate_metrics(type_targets, type_preds, type_probs)
    
    # 准备绘图数据
    datasets = ['Overall'] + unique_types
    metrics_to_plot = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
    
    data_for_plot = {metric: [] for metric in metrics_to_plot}
    
    # 整体指标
    data_for_plot['AUC'].append(overall_metrics['auc'])
    data_for_plot['Accuracy'].append(overall_metrics['accuracy'])
    data_for_plot['Sensitivity'].append(overall_metrics['sensitivity'])
    data_for_plot['Specificity'].append(overall_metrics['specificity'])
    data_for_plot['PPV'].append(overall_metrics['ppv'])
    data_for_plot['NPV'].append(overall_metrics['npv'])
    
    # 各type指标
    for data_type in unique_types:
        metrics = type_metrics[data_type]
        data_for_plot['AUC'].append(metrics['auc'] if not np.isnan(metrics['auc']) else 0)
        data_for_plot['Accuracy'].append(metrics['accuracy'])
        data_for_plot['Sensitivity'].append(metrics['sensitivity'])
        data_for_plot['Specificity'].append(metrics['specificity'])
        data_for_plot['PPV'].append(metrics['ppv'])
        data_for_plot['NPV'].append(metrics['npv'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('独立验证集性能对比 - V75模型', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 为每个指标绘制柱状图
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        x_pos = np.arange(len(datasets))
        bars = ax.bar(x_pos, data_for_plot[metric], color=colors[:len(datasets)], alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, data_for_plot[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(datasets, rotation=15, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 性能对比图已保存到: {output_path}")

def generate_report(all_results, output_path):
    """生成详细的验证报告"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("独立验证集推理报告 - V75模型")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 整体指标
    report_lines.append("📊 整体验证集性能:")
    report_lines.append(f"  总样本数: {len(all_results['bom_targets'])}")
    
    overall_metrics = calculate_metrics(
        all_results['bom_targets'],
        all_results['bom_preds'],
        all_results['bom_probs']
    )
    
    report_lines.append(f"  良性样本: {overall_metrics['negative']} ({overall_metrics['negative']/overall_metrics['total']*100:.1f}%)")
    report_lines.append(f"  恶性样本: {overall_metrics['positive']} ({overall_metrics['positive']/overall_metrics['total']*100:.1f}%)")
    report_lines.append("")
    report_lines.append("  性能指标:")
    report_lines.append(f"    AUC:         {overall_metrics['auc']:.4f}")
    report_lines.append(f"    Accuracy:    {overall_metrics['accuracy']:.4f}")
    report_lines.append(f"    Sensitivity: {overall_metrics['sensitivity']:.4f}")
    report_lines.append(f"    Specificity: {overall_metrics['specificity']:.4f}")
    report_lines.append(f"    PPV:         {overall_metrics['ppv']:.4f}")
    report_lines.append(f"    NPV:         {overall_metrics['npv']:.4f}")
    report_lines.append("")
    report_lines.append("  混淆矩阵:")
    report_lines.append(f"    TP (真阳性): {overall_metrics['tp']}")
    report_lines.append(f"    TN (真阴性): {overall_metrics['tn']}")
    report_lines.append(f"    FP (假阳性): {overall_metrics['fp']}")
    report_lines.append(f"    FN (假阴性): {overall_metrics['fn']}")
    report_lines.append("")
    
    # 按type分组统计
    report_lines.append("=" * 80)
    report_lines.append("📊 按数据来源分组的性能分析:")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 获取所有type
    unique_types = sorted(set(all_results['types']))
    
    for data_type in unique_types:
        # 筛选该type的数据
        type_mask = np.array([t == data_type for t in all_results['types']])
        type_targets = np.array(all_results['bom_targets'])[type_mask]
        type_preds = np.array(all_results['bom_preds'])[type_mask]
        type_probs = np.array(all_results['bom_probs'])[type_mask]
        
        report_lines.append(f"🔍 Type: {data_type}")
        report_lines.append(f"  样本数: {len(type_targets)}")
        
        if len(type_targets) > 0:
            type_metrics = calculate_metrics(type_targets, type_preds, type_probs)
            
            report_lines.append(f"  良性样本: {type_metrics['negative']} ({type_metrics['negative']/type_metrics['total']*100:.1f}%)")
            report_lines.append(f"  恶性样本: {type_metrics['positive']} ({type_metrics['positive']/type_metrics['total']*100:.1f}%)")
            report_lines.append("")
            report_lines.append("  性能指标:")
            
            auc_str = f"{type_metrics['auc']:.4f}" if not np.isnan(type_metrics['auc']) else "N/A (单类别)"
            report_lines.append(f"    AUC:         {auc_str}")
            report_lines.append(f"    Accuracy:    {type_metrics['accuracy']:.4f}")
            report_lines.append(f"    Sensitivity: {type_metrics['sensitivity']:.4f}")
            report_lines.append(f"    Specificity: {type_metrics['specificity']:.4f}")
            report_lines.append(f"    PPV:         {type_metrics['ppv']:.4f}")
            report_lines.append(f"    NPV:         {type_metrics['npv']:.4f}")
            report_lines.append("")
            report_lines.append("  混淆矩阵:")
            report_lines.append(f"    TP: {type_metrics['tp']}, TN: {type_metrics['tn']}, FP: {type_metrics['fp']}, FN: {type_metrics['fn']}")
        else:
            report_lines.append("  无有效样本")
        
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("")
    
    # 写入文件
    report_content = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 同时打印到控制台
    print(report_content)
    print(f"\n📄 报告已保存到: {output_path}")

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 80)
    print("独立验证集推理 - V75模型")
    print("=" * 80)
    
    device = get_device()
    print(f"🔬 设备: {device}")
    print(f"📁 模型路径: {CONFIG['model_path']}")
    print(f"📊 验证数据: {CONFIG['verify_data']}")
    print(f"🖼️ 图像根目录: {CONFIG['verify_root']}")
    print()
    
    # 检查模型文件
    if not os.path.exists(CONFIG['model_path']):
        print(f"❌ 错误: 模型文件不存在: {CONFIG['model_path']}")
        return
    
    # 加载特征映射
    with open(CONFIG['feature_mapping_file'], 'r', encoding='utf-8') as f:
        feature_mapping = json.load(f)
    print(f"✅ 加载特征映射: {list(feature_mapping.keys())}")
    
    # 加载验证数据
    df_verify_original = pd.read_csv(CONFIG['verify_data'])
    print(f"✅ 加载验证数据: {len(df_verify_original)} 样本")
    print(f"   Type分布: {df_verify_original['type'].value_counts().to_dict()}")
    print()
    
    # 过滤缺失图像
    df_verify_filtered, missing_samples = filter_missing_images(df_verify_original, CONFIG['verify_root'])
    print()
    
    # 如果缺失太多，给出警告
    if len(missing_samples) > 0:
        missing_ratio = len(missing_samples) / len(df_verify_original) * 100
        if missing_ratio > 5:
            print(f"⚠️⚠️⚠️ 警告: 缺失图像比例较高 ({missing_ratio:.2f}%)，可能影响评估准确性！")
            print()
    
    # 使用过滤后的数据集进行推理
    df_verify = df_verify_filtered
    
    # 创建数据集和加载器
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    verify_dataset = NoduleFeatureDataset(
        df_verify,
        CONFIG['verify_root'],
        feature_mapping,
        val_transform
    )
    
    verify_loader = DataLoader(
        verify_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )
    
    print(f"✅ 创建数据集: {len(verify_dataset)} 样本")
    print()
    
    # 加载模型
    print("🔄 加载模型...")
    model = MultiTaskNoduleCNN(feature_mapping, dropout_rate=CONFIG['dropout_rate']).to(device)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    print("✅ 模型加载成功")
    print()
    
    # 推理
    print("🚀 开始推理...")
    print(f"⚙️  良恶性判断阈值: {CONFIG['bom_threshold']} (恶性概率 >= 阈值 → 判为恶性)")
    all_results = inference_and_collect(model, verify_loader, device, threshold=CONFIG['bom_threshold'])
    print(f"✅ 推理完成: 收集 {len(all_results['bom_targets'])} 个有效样本")
    print()
    
    # 生成报告
    print("📝 生成验证报告...")
    generate_report(all_results, CONFIG['output_report'])
    
    # 生成性能对比图
    print("\n📊 生成性能对比图...")
    plot_path = CONFIG['output_report'].replace('.txt', '_comparison.png')
    plot_performance_comparison(all_results, plot_path)
    
    # 保存预测结果到CSV
    print("\n💾 保存预测结果到CSV...")
    save_predictions_to_csv(
        df_verify_original, 
        df_verify_filtered, 
        all_results, 
        CONFIG['output_csv'],
        missing_samples
    )

if __name__ == "__main__":
    main()
