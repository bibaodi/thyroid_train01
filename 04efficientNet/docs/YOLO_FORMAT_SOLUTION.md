# YOLO Classification Format - Complete Solution

## Problem
YOLO classification training failed with:
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**Root Cause**: YOLO expects a dataset directory with `train/`, `val/`, and/or `test/` subdirectories, each containing class folders.

## Solution Implemented

### 1. Updated Function: `save_images_in_yolo_format()`

**New Parameters:**
- `split_name`: Specify 'train', 'val', or 'test'
- `val_split`: Fraction for validation split (0.0-1.0)

**Key Features:**
- ✅ Creates proper YOLO directory structure with train/val/test splits
- ✅ Automatic stratified train/val split when `val_split > 0`
- ✅ Maintains class balance in splits
- ✅ Detailed statistics per split and class

### 2. Directory Structure Created

```
/opt/dlami/nvme/yolo_format_dataset/
├── train/
│   ├── class_0/  (Benign - 80% of data)
│   │   ├── <sop_uid_1>.jpg
│   │   ├── <sop_uid_2>.jpg
│   │   └── ...
│   └── class_1/  (Malignant - 80% of data)
│       ├── <sop_uid_3>.jpg
│       └── ...
├── val/
│   ├── class_0/  (Benign - 20% of data)
│   └── class_1/  (Malignant - 20% of data)
└── test/
    ├── class_0/  (Benign - verification data)
    └── class_1/  (Malignant - verification data)
```

### 3. Integration in Training Pipeline

**Training Data:**
```python
save_images_in_yolo_format(
    df_train_filtered, 
    train_dataImgRoot, 
    yolo_output_dir='/opt/dlami/nvme/yolo_format_dataset', 
    class_column='bom', 
    df_name="Training Data",
    split_name='train',
    val_split=0.2  # Creates both train/ and val/ subdirectories
)
```

**Verification Data:**
```python
save_images_in_yolo_format(
    df_verify_filtered, 
    verify_image_root, 
    yolo_output_dir='/opt/dlami/nvme/yolo_format_dataset', 
    class_column='bom', 
    df_name="Verification Data",
    split_name='test',
    val_split=0.0  # No split, saves directly to test/
)
```

## Module Structure

The YOLO dataset utilities have been extracted to a separate module:

```
srcs/
├── yolo_dataset_utils.py          # YOLO dataset utilities (NEW)
└── train_nodule_feature_cnn_model_v75.py  # Main training script
```

**Import in your code:**
```python
from yolo_dataset_utils import save_images_in_yolo_format
```

## Usage in YOLO Training

### Option 1: Use the complete dataset (with train/val/test)
```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

# Point to the root directory containing train/, val/, test/
datasetf = r'/opt/dlami/nvme/yolo_format_dataset'

results = model.train(
    data=datasetf,
    epochs=90,
    imgsz=96,
    erasing=0.2
)
```

### Option 2: Use only train split (YOLO will look for val/)
```python
# YOLO will automatically find train/ and val/ subdirectories
datasetf = r'/opt/dlami/nvme/yolo_format_dataset'
results = model.train(data=datasetf, epochs=90, imgsz=96)
```

### Option 3: Specify splits explicitly
```python
# If you want to use different splits
train_path = r'/opt/dlami/nvme/yolo_format_dataset/train'
val_path = r'/opt/dlami/nvme/yolo_format_dataset/val'

results = model.train(
    data=train_path,
    val=val_path,
    epochs=90,
    imgsz=96
)
```

## Statistics Output Example

```
💾 Saving images in YOLO classification format for Training Data...
  - Source: /data/dataInTrain/251016-efficientNet/dataset_images/nodule_images
  - Target: /opt/dlami/nvme/yolo_format_dataset
  - Class column: bom
  - Split: train
  - Found 2 unique classes: [0, 1]
  - Splitting data: 80% train, 20% val

  📂 Processing train split (8000 samples)...
  📂 Processing val split (2000 samples)...

  📊 Overall Save Statistics:
    - Total images processed: 10000
    - Successfully copied: 9950
    - Failed: 50
    - Success rate: 99.50%

  📁 Images by split and class:
    TRAIN:
      Total: 7960/8000
      - Class 0 (Benign): 5572 images
      - Class 1 (Malignant): 2388 images
    VAL:
      Total: 1990/2000
      - Class 0 (Benign): 1393 images
      - Class 1 (Malignant): 597 images

  ✅ YOLO format dataset saved to: /opt/dlami/nvme/yolo_format_dataset
```

## Key Improvements

1. ✅ **Proper YOLO Structure**: Creates train/val/test subdirectories
2. ✅ **Automatic Splitting**: Stratified split maintains class balance
3. ✅ **No Data Leakage**: Separate train/val/test splits
4. ✅ **Detailed Logging**: Per-split and per-class statistics
5. ✅ **Flexible**: Can create any split (train/val/test) independently
6. ✅ **Error Handling**: Graceful handling of missing images

## Troubleshooting

### Error: "No such file or directory"
- Ensure the output directory path is valid
- Check that source images exist in `image_root`

### Error: "No class directories found"
- Verify that the DataFrame has valid BOM labels (0 or 1)
- Check that images were successfully copied

### YOLO still can't find dataset
- Verify directory structure: `ls -la /opt/dlami/nvme/yolo_format_dataset/`
- Should see: `train/`, `val/`, `test/` directories
- Each should contain: `class_0/`, `class_1/` subdirectories

## Summary

The updated function now creates a complete YOLO-compatible dataset structure with proper train/val/test splits, solving the original error and providing a production-ready solution for YOLO classification training.
