#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Dataset Utilities
Utilities for saving datasets in YOLO classification format.
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def save_images_in_yolo_format(df, image_root, output_root, class_column='bom', df_name="Dataset", split_name='train', val_split=0.2):
    """
    Save images in YOLO classification format organized by class labels.
    
    YOLO classification format structure:
    output_root/
        ├── train/
        │   ├── class_0/
        │   │   ├── image1.jpg
        │   │   └── image2.jpg
        │   └── class_1/
        │       ├── image3.jpg
        │       └── image4.jpg
        └── val/
            ├── class_0/
            └── class_1/
    
    Args:
        df: DataFrame with 'access_no', 'sop_uid', and class_column
        image_root: Root directory where source images are located
        output_root: Root directory where YOLO format images will be saved
        class_column: Column name containing class labels (default: 'bom')
        df_name: Name of the dataset for logging
        split_name: Name of the split ('train', 'val', 'test')
        val_split: Fraction of data to use for validation (0.0-1.0). If 0, no split is performed.
    
    Returns:
        Dictionary with statistics about saved images
    """
    print(f"\n💾 Saving images in YOLO classification format for {df_name}...")
    print(f"  - Source: {image_root}")
    print(f"  - Target: {output_root}")
    print(f"  - Class column: {class_column}")
    print(f"  - Split: {split_name}")
    
    if class_column not in df.columns:
        print(f"  ⚠️  WARNING: Column '{class_column}' not found in DataFrame")
        return {}
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Statistics
    stats = {
        'total': 0,
        'copied': 0,
        'failed': 0,
        'by_split': {},
        'by_class': {}
    }
    
    # Get unique classes
    unique_classes = df[class_column].dropna().unique()
    print(f"  - Found {len(unique_classes)} unique classes: {sorted(unique_classes)}")
    
    # Split data into train/val if val_split > 0 and split_name is 'train'
    splits_to_process = [(split_name, df)]
    
    if val_split > 0 and split_name == 'train':
        print(f"  - Splitting data: {(1-val_split)*100:.0f}% train, {val_split*100:.0f}% val")
        # Stratified split to maintain class balance
        train_df, val_df = train_test_split(
            df, 
            test_size=val_split, 
            stratify=df[class_column].dropna(),
            random_state=42
        )
        splits_to_process = [('train', train_df), ('val', val_df)]
        stats['by_split']['train'] = {'total': 0, 'copied': 0, 'by_class': {}}
        stats['by_split']['val'] = {'total': 0, 'copied': 0, 'by_class': {}}
    else:
        stats['by_split'][split_name] = {'total': 0, 'copied': 0, 'by_class': {}}
    
    # Process each split
    for current_split, split_df in splits_to_process:
        print(f"\n  📂 Processing {current_split} split ({len(split_df)} samples)...")
        
        # Create split directory
        split_dir = os.path.join(output_root, current_split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class directories for this split
        for class_label in unique_classes:
            class_dir = os.path.join(split_dir, f"class_{int(class_label)}")
            os.makedirs(class_dir, exist_ok=True)
            if int(class_label) not in stats['by_split'][current_split]['by_class']:
                stats['by_split'][current_split]['by_class'][int(class_label)] = 0
        
        # Copy images to respective class folders
        for idx, row in split_df.iterrows():
            try:
                access_no = str(row['access_no'])
                sop_uid = str(row['sop_uid'])
                class_label = row[class_column]
                
                # Skip if class label is NaN
                if pd.isna(class_label):
                    continue
                
                stats['total'] += 1
                stats['by_split'][current_split]['total'] += 1
                
                # Source image path
                src_path = os.path.join(image_root, access_no, f"{sop_uid}.jpg")
                
                # Target image path - use sop_uid as filename to maintain uniqueness
                class_dir = os.path.join(split_dir, f"class_{int(class_label)}")
                dst_path = os.path.join(class_dir, f"{sop_uid}.jpg")
                
                # Copy image
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    stats['copied'] += 1
                    stats['by_split'][current_split]['copied'] += 1
                    stats['by_split'][current_split]['by_class'][int(class_label)] += 1
                    
                    # Update global class stats
                    if int(class_label) not in stats['by_class']:
                        stats['by_class'][int(class_label)] = 0
                    stats['by_class'][int(class_label)] += 1
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                print(f"  ⚠️  Error processing row {idx}: {e}")
                stats['failed'] += 1
                continue
    
    # Print statistics
    print(f"\n  📊 Overall Save Statistics:")
    print(f"    - Total images processed: {stats['total']}")
    print(f"    - Successfully copied: {stats['copied']}")
    print(f"    - Failed: {stats['failed']}")
    print(f"    - Success rate: {stats['copied']/stats['total']*100:.2f}%" if stats['total'] > 0 else "    - Success rate: N/A")
    
    print(f"\n  📁 Images by split and class:")
    for split in stats['by_split']:
        split_stats = stats['by_split'][split]
        print(f"    {split.upper()}:")
        print(f"      Total: {split_stats['copied']}/{split_stats['total']}")
        for class_label in sorted(split_stats['by_class'].keys()):
            count = split_stats['by_class'][class_label]
            class_name = "Benign" if class_label == 0 else "Malignant"
            print(f"      - Class {class_label} ({class_name}): {count} images")
    
    print(f"\n  ✅ YOLO format dataset saved to: {output_root}")
    
    return stats
