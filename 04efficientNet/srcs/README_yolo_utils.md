# YOLO Dataset Utilities

Utilities for converting image datasets to YOLO classification format.

## Module: `yolo_dataset_utils.py`

### Function: `save_images_in_yolo_format()`

Converts a pandas DataFrame with image metadata into YOLO classification format with proper directory structure.

#### Parameters

- **df** (DataFrame): DataFrame containing image metadata with columns:
  - `access_no`: Patient/study identifier
  - `sop_uid`: Unique image identifier
  - `{class_column}`: Classification label (e.g., 'bom' with values 0 or 1)

- **image_root** (str): Root directory where source images are located
  - Expected structure: `{image_root}/{access_no}/{sop_uid}.jpg`

- **output_root** (str): Target directory for YOLO format dataset

- **class_column** (str, default='bom'): Column name containing class labels

- **df_name** (str, default='Dataset'): Dataset name for logging

- **split_name** (str, default='train'): Split name ('train', 'val', or 'test')

- **val_split** (float, default=0.2): Validation split ratio (0.0-1.0)
  - If > 0 and split_name='train': automatically creates train/val split
  - If 0: saves all data to the specified split_name

#### Returns

Dictionary with statistics:
```python
{
    'total': int,           # Total images processed
    'copied': int,          # Successfully copied images
    'failed': int,          # Failed copies
    'by_split': {           # Per-split statistics
        'train': {
            'total': int,
            'copied': int,
            'by_class': {0: int, 1: int}
        },
        'val': {...}
    },
    'by_class': {0: int, 1: int}  # Overall class distribution
}
```

#### Output Structure

```
output_root/
├── train/
│   ├── class_0/
│   │   ├── {sop_uid_1}.jpg
│   │   ├── {sop_uid_2}.jpg
│   │   └── ...
│   └── class_1/
│       ├── {sop_uid_3}.jpg
│       └── ...
├── val/
│   ├── class_0/
│   └── class_1/
└── test/
    ├── class_0/
    └── class_1/
```

## Usage Examples

### Example 1: Create train/val split automatically

```python
from yolo_dataset_utils import save_images_in_yolo_format
import pandas as pd

# Load your data
df = pd.read_csv('dataset.csv')

# Save with automatic 80/20 train/val split
stats = save_images_in_yolo_format(
    df=df,
    image_root='/path/to/source/images',
    output_root='/path/to/yolo_dataset',
    class_column='bom',
    df_name='Training Data',
    split_name='train',
    val_split=0.2  # Creates both train/ and val/
)

print(f"Copied {stats['copied']} images")
```

### Example 2: Create test split without splitting

```python
# Save verification data as test split
stats = save_images_in_yolo_format(
    df=df_verify,
    image_root='/path/to/verify/images',
    output_root='/path/to/yolo_dataset',
    class_column='bom',
    df_name='Verification Data',
    split_name='test',
    val_split=0.0  # No split, saves directly to test/
)
```

### Example 3: Use in YOLO training

```python
from ultralytics import YOLO

# After creating the dataset
model = YOLO('yolov8n-cls.pt')

results = model.train(
    data='/path/to/yolo_dataset',  # Points to root with train/val/test
    epochs=90,
    imgsz=96
)
```

## Features

✅ **Automatic stratified splitting** - Maintains class balance in train/val splits  
✅ **Flexible split configuration** - Create train, val, or test splits independently  
✅ **Detailed statistics** - Per-split and per-class reporting  
✅ **Error handling** - Graceful handling of missing images  
✅ **YOLO compatible** - Creates proper directory structure for YOLO classification  
✅ **Unique filenames** - Uses sop_uid to prevent filename collisions  

## Integration

This module is used in `train_nodule_feature_cnn_model_v75.py`:

```python
from yolo_dataset_utils import save_images_in_yolo_format

# Called after validate_and_filter_images()
save_images_in_yolo_format(
    df_train_filtered,
    train_dataImgRoot,
    yolo_output_dir,
    class_column='bom',
    split_name='train',
    val_split=0.2
)
```

## Dependencies

- pandas
- scikit-learn (for train_test_split)
- os, shutil (standard library)

## Notes

- Images are copied (not moved) to preserve originals
- Class labels are converted to `class_{label}` directory names
- Stratified splitting ensures balanced class distribution
- Missing images are logged but don't stop the process
