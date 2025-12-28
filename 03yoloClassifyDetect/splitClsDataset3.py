import os
import random
import argparse
from pathlib import Path

def split_dataset(origin_path, split_ratio, output_name):
    # Convert relative paths to absolute paths
    origin_path = Path(origin_path).resolve()
    outpath = Path(output_name).resolve()
    if False == outpath.is_dir():
        output_path = origin_path.parent / output_name
    else:
        output_path = outpath
    print(f">>>outPath={output_path}")

    # Check if the origin path exists
    if not origin_path.exists() or not origin_path.is_dir():
        raise ValueError(f"The origin path '{origin_path}' does not exist or is not a directory.")

    # Check if the split ratio is valid
    if not (0 < split_ratio < 1):
        raise ValueError("The split ratio must be a float between 0 and 1.")

    # Create the necessary directories
    train_path = output_path / 'train'
    val_path = output_path / 'val'

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    # Iterate over each class directory
    for class_dir in origin_path.iterdir():
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

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument("origin_path", type=str, help="Path to the original dataset directory.")
    parser.add_argument("split_ratio", type=float, help="Split ratio for the training set (e.g., 0.8 for 80% train).")
    parser.add_argument("output_name", type=str, help="Name of the output dataset directory.")

    args = parser.parse_args()

    split_dataset(args.origin_path, args.split_ratio, args.output_name)

if __name__ == "__main__":
    main()

