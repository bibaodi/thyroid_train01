#!/usr/bin/env python3
"""
将图像数据集预处理成LMDB格式，大幅提升I/O性能
"""
import os
import sys
import json
import lmdb
import pickle
import pandas as pd
from PIL import Image
from tqdm import tqdm
import io
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def apply_filters(df, config):
    """应用与训练相同的过滤规则"""
    from train_nodule_feature_cnn_model_v75 import apply_all_filters
    return apply_all_filters(df, config, df_name="LMDB Data")

def create_lmdb_dataset(csv_path, image_root, output_path, map_size_gb=50, apply_training_filters=False):
    """
    将图像数据集转换为LMDB格式
    
    Args:
        csv_path: CSV数据文件路径
        image_root: 图像根目录
        output_path: LMDB输出路径
        map_size_gb: LMDB最大大小(GB)
        apply_training_filters: 是否应用训练时的过滤规则
    """
    print(f"📊 加载CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"   共 {len(df)} 条记录")
    
    # 应用训练时的过滤规则
    if apply_training_filters:
        print(f"\n🔍 应用训练过滤规则...")
        from train_nodule_feature_cnn_model_v75 import CONFIG
        df = apply_filters(df, CONFIG)
        print(f"   过滤后: {len(df)} 条记录")
    
    # 创建LMDB
    map_size = map_size_gb * 1024 * 1024 * 1024  # GB to bytes
    env = lmdb.open(output_path, map_size=map_size)
    
    success_count = 0
    fail_count = 0
    
    print(f"\n🔄 开始处理图像...")
    with env.begin(write=True) as txn:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            access_no = row['access_no']
            sop_uid = row['sop_uid']
            image_path = os.path.join(image_root, access_no, f"{sop_uid}.jpg")
            
            try:
                # 读取图像为字节
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                
                # 准备数据
                data = {
                    'image': image_bytes,
                    'access_no': access_no,
                    'sop_uid': sop_uid,
                    'bom': row.get('bom'),
                    'ti_rads': row.get('ti_rads'),
                    'composition': row.get('composition'),
                    'echo': row.get('echo'),
                    'foci': row.get('foci'),
                    'margin': row.get('margin'),
                    'shape': row.get('shape'),
                }
                
                # 存储
                key = f"{idx}".encode()
                value = pickle.dumps(data)
                txn.put(key, value)
                success_count += 1
                
            except FileNotFoundError:
                fail_count += 1
            except Exception as e:
                fail_count += 1
                if fail_count <= 5:
                    print(f"   Error at {idx}: {e}")
        
        # 存储元数据
        meta = {
            'num_samples': success_count,
            'csv_path': csv_path,
            'image_root': image_root,
        }
        txn.put(b'__meta__', pickle.dumps(meta))
    
    env.close()
    
    print(f"\n✅ LMDB创建完成!")
    print(f"   成功: {success_count}")
    print(f"   失败: {fail_count}")
    print(f"   输出: {output_path}")
    
    # 检查文件大小
    total_size = 0
    for f in os.listdir(output_path):
        total_size += os.path.getsize(os.path.join(output_path, f))
    print(f"   大小: {total_size / 1024 / 1024 / 1024:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description='创建LMDB数据集')
    parser.add_argument('--csv', default='data/dataset_table/train/all_matched_sops_ds_v3_with_tr13_0926_with_OOF_suspect.csv')
    parser.add_argument('--image_root', default='data/dataset_images/2nodule_images')
    parser.add_argument('--output', default='data/dataset_lmdb/train_lmdb')
    parser.add_argument('--map_size', type=int, default=50, help='LMDB max size in GB')
    parser.add_argument('--filter', action='store_true', help='Apply training filters')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    create_lmdb_dataset(args.csv, args.image_root, args.output, args.map_size, args.filter)

if __name__ == '__main__':
    main()

