import os
import random
import argparse
import logging
from pathlib import Path

def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file),
                                  logging.StreamHandler()])

def split_dataset(input_path, split_ratio, output_path):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    images_dir = input_path / 'images'
    labels_dir = input_path / 'labels'

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Images or labels directory not found in {input_path}")

    train_images_dir = output_path / 'images' / 'train'
    val_images_dir = output_path / 'images' / 'val'
    train_labels_dir = output_path / 'labels' / 'train'
    val_labels_dir = output_path / 'labels' / 'val'

    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob('*.jpg'))
    random.shuffle(image_files)

    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    def create_symlinks(file_list, img_dest, lbl_dest):
        for img_file in file_list:
            label_file = labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                os.symlink(img_file, img_dest / img_file.name)
                os.symlink(label_file, lbl_dest / label_file.name)
                logging.info(f"Created symlink for {img_file.name} and {label_file.name} in {img_dest.name} and {lbl_dest.name}")
            else:
                logging.warning(f"Label file for {img_file.name} not found.")

    create_symlinks(train_files, train_images_dir, train_labels_dir)
    create_symlinks(val_files, val_images_dir, val_labels_dir)

    logging.info("Dataset split complete with symlinks.")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets with symlinks.")
    parser.add_argument("input_path", type=str, help="Path to the original dataset directory.")
    parser.add_argument("split_ratio", type=float, help="Split ratio for the training set (e.g., 0.8 for 80% train).")
    parser.add_argument("output_path", type=str, help="Path to the output dataset directory.")

    args = parser.parse_args()

    log_file = Path(args.input_path) / 'split_dataset.log'
    setup_logging(log_file)

    split_dataset(args.input_path, args.split_ratio, args.output_path)

if __name__ == "__main__":
    main()

