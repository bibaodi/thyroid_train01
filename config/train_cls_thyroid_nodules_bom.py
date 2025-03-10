import os
import cv2
import json
import random
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from model_test.contour_json_handler import ContourJsonHandler


def prepare_data(src_path: str, dataset_dir: str):
    """准备训练数据集
    去掉了裁剪到 nodules 的功能，全图参与训练
    Args:
        src_path: 源数据目录
        dataset_dir: 数据集输出目录
    """
    # 检查数据集是否已准备完成
    if check_dataset(dataset_dir):
        print(f"\n数据集已存在且完整，跳过数据准备阶段...")
        return

    print(f"\n开始准备数据集...")
    # 创建数据集目录结构
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    # 为每个类别创建目录
    for class_id in [0, 1]:  # 0: negative, 1: positive
        os.makedirs(os.path.join(train_dir, str(class_id)), exist_ok=True)
        os.makedirs(os.path.join(val_dir, str(class_id)), exist_ok=True)

    # 处理数据集
    processed_data = []

    # 处理阴性和阳性数据
    for class_id, class_name in enumerate(['negative', 'positive']):
        class_dir = os.path.join(src_path, str(class_id))
        if not os.path.exists(class_dir):
            continue

        print(f"\n处理{class_name}数据...")
        for root, _, files in os.walk(class_dir):
            for file in files:
                if not file.endswith(('.jpg', '.png', '.bmp')):
                    continue

                image_path = os.path.join(root, file)
                json_path = os.path.splitext(image_path)[0] + '.json'
                if not os.path.exists(json_path):
                    continue

                try:
                    # 读取图像
                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    processed_data.append({
                        'image': image,
                        'class_id': class_id,
                        'original_file': file
                    })

                except Exception as e:
                    print(f"\n处理文件 {file} 时出错: {e}")
                    continue

    # 随机打乱数据
    random.seed(42)
    random.shuffle(processed_data)

    # 分割数据集 (80% 训练, 20% 验证)
    train_size = int(len(processed_data) * 0.8)

    # 保存数据集
    for idx, item in enumerate(processed_data):
        if idx < train_size:
            save_dir = os.path.join(train_dir, str(item['class_id']))
        else:
            save_dir = os.path.join(val_dir, str(item['class_id']))

        # 生成唯一文件名，确保只保存图像文件
        output_filename = f"{os.path.splitext(item['original_file'])[0]}_{idx}.jpg"
        output_path = os.path.join(save_dir, output_filename)

        # 保存图像文件
        cv2.imwrite(output_path, item['image'])

        # 删除可能存在的.npy文件
        npy_path = os.path.splitext(output_path)[0] + '.npy'
        if os.path.exists(npy_path):
            os.remove(npy_path)

    print(f"\n数据集处理完成:")
    print(f"- 总样本数: {len(processed_data)}")
    print(f"- 训练集数量: {train_size}")
    print(f"- 验证集数量: {len(processed_data) - train_size}")
    print(f"数据集保存在: {dataset_dir}")

def check_dataset(dataset_dir: str) -> bool:
    """检查数据集是否已存在且完整"""
    # 检查必要的目录结构
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    for dir_path in [train_dir, val_dir]:
        for class_id in ['0', '1']:
            class_dir = os.path.join(dir_path, class_id)
            if not os.path.exists(class_dir):
                return False
            # 检查每个类别目录是否包含图像文件
            if not any(f.endswith(('.jpg', '.png', '.bmp')) for f in os.listdir(class_dir)):
                return False
    return True

def train(data_yaml: str, weights: str = 'yolov8n-cls.pt', epochs: int = 100):
    """训练模型
    Args:
        data_yaml: 数据配置文件路径
        weights: 预训练权重文件路径
        epochs: 训练轮数
    """
    # 加载YAML配置
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # 获取训练参数
    train_name = "thyroid_nodules_bom_all_v2"  # 训练名称和版本
    augment_cfg = config.get('augment', {})

    weights_path = config.get('model', weights)
    
    # 加载模型
    model = YOLO(weights_path)

    # 开始训练
    model.train(
        data=config['path'],  # 使用YAML中配置的数据集路径
        #data=data_yaml,
        epochs=epochs,
        imgsz=None, # 引入尺寸信息
        batch=augment_cfg.get('batch', 32),
        name=train_name,
        device='mps',
        task='classify',  # 明确指定任务类型为分类
        # 应用数据增强参数
        #hsv_h=augment_cfg.get('hsv_h', 0.0),
        #hsv_s=augment_cfg.get('hsv_s', 0.0),
        hsv_v=augment_cfg.get('hsv_v', 0.4),
        #degrees=augment_cfg.get('degrees', 0.0),
        #translate=augment_cfg.get('translate', 0.1),
        scale=augment_cfg.get('scale', 0.5),
        #shear=augment_cfg.get('shear', 0.0),
        #perspective=augment_cfg.get('perspective', 0.0),
        #flipud=augment_cfg.get('flipud', 0.0),
        #fliplr=augment_cfg.get('fliplr', 0.5),
        #mosaic=augment_cfg.get('mosaic', 0.3),
        #mixup=augment_cfg.get('mixup', 0.0),
        #copy_paste=augment_cfg.get('copy_paste', 0.0)
    )

def main():
    # 设置路径
    base_dir = Path(__file__).parent
    data_yaml = base_dir / 'thyroid_nodules_bom.yaml'
    
    # 从YAML文件加载配置
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # 从配置中读取路径
    src_path = config.get('src_path')
    dataset_dir = config.get('path')  # 使用YAML中的path字段作为dataset_dir

    # 准备数据集
    prepare_data(str(src_path), str(dataset_dir))

    # 训练模型
    train(str(data_yaml))

if __name__ == '__main__':
    main()