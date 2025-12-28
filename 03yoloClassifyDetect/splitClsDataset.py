import os
import random
from pathlib import Path

def split_dataset(origin_path, split_ratio, output_name):
    # Define the output directory
    output_path = Path(origin_path).parent / output_name

    # Create the necessary directories
    train_path = output_path / 'train'
    val_path = output_path / 'val'

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    # Iterate over each class directory
    for class_dir in Path(origin_path).iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            files = list(class_dir.iterdir())
            random.shuffle(files)

            # Split the files into train and val sets
            split_index = int(len(files) * split_ratio)
            train_files = files[:split_index]
            val_files = files[split_index:]

            # Create class directories in train and val
            (train_path / class_name).mkdir(exist_ok=True)
            (val_path / class_name).mkdir(exist_ok=True)

            # Create symbolic links
            for file in train_files:
                link_path = train_path / class_name / file.name
                link_path.symlink_to(file)

            for file in val_files:
                link_path = val_path / class_name / file.name
                link_path.symlink_to(file)

    print(f"Dataset split complete. Output directory: {output_path}")

# Example usage
origin_dataset_path = 'dataset'
split_ratio = 0.8
output_dataset_name = 'datasetB'

split_dataset(origin_dataset_path, split_ratio, output_dataset_name)

