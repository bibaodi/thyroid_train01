# Quick Reference: YOLO Dataset Utilities

## Import

```python
from yolo_dataset_utils import save_images_in_yolo_format
```

## Basic Usage

```python
save_images_in_yolo_format(
    df=your_dataframe,
    image_root='/path/to/source/images',
    output_root='/path/to/yolo_dataset',
    class_column='bom',
    df_name='My Dataset',
    split_name='train',
    val_split=0.2
)
```

## Common Scenarios

### Scenario 1: Training data with auto train/val split
```python
save_images_in_yolo_format(
    df=df_train,
    image_root='/data/images',
    output_root='/output/yolo_dataset',
    split_name='train',
    val_split=0.2  # 80% train, 20% val
)
# Creates: /output/yolo_dataset/train/ and /output/yolo_dataset/val/
```

### Scenario 2: Test data without splitting
```python
save_images_in_yolo_format(
    df=df_test,
    image_root='/data/images',
    output_root='/output/yolo_dataset',
    split_name='test',
    val_split=0.0  # No split
)
# Creates: /output/yolo_dataset/test/
```

### Scenario 3: Custom class column
```python
save_images_in_yolo_format(
    df=df,
    image_root='/data/images',
    output_root='/output/yolo_dataset',
    class_column='diagnosis',  # Instead of 'bom'
    split_name='train',
    val_split=0.2
)
```

## Expected DataFrame Format

Your DataFrame should have these columns:
- `access_no`: Patient/study identifier
- `sop_uid`: Unique image identifier  
- `{class_column}`: Classification label (e.g., 0 or 1)

Example:
```
   access_no              sop_uid                    bom
0  PATIENT001             1.2.840.113619.xxx         0
1  PATIENT001             1.2.840.113619.yyy         0
2  PATIENT002             1.2.840.113619.zzz         1
```

## Expected Source Image Structure

```
image_root/
├── PATIENT001/
│   ├── 1.2.840.113619.xxx.jpg
│   └── 1.2.840.113619.yyy.jpg
└── PATIENT002/
    └── 1.2.840.113619.zzz.jpg
```

## Output Structure

```
output_root/
├── train/
│   ├── class_0/
│   │   └── 1.2.840.113619.xxx.jpg
│   └── class_1/
│       └── 1.2.840.113619.zzz.jpg
└── val/
    ├── class_0/
    └── class_1/
```

## Use with YOLO

```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')
results = model.train(
    data='/output/yolo_dataset',  # Root directory
    epochs=90,
    imgsz=96
)
```

## Return Value

```python
stats = save_images_in_yolo_format(...)

# Access statistics
print(f"Total: {stats['total']}")
print(f"Copied: {stats['copied']}")
print(f"Failed: {stats['failed']}")
print(f"Train images: {stats['by_split']['train']['copied']}")
print(f"Val images: {stats['by_split']['val']['copied']}")
print(f"Class 0: {stats['by_class'][0]}")
print(f"Class 1: {stats['by_class'][1]}")
```

## Troubleshooting

### Import Error
```python
# If import fails, add srcs to path:
import sys
sys.path.insert(0, 'path/to/srcs')
from yolo_dataset_utils import save_images_in_yolo_format
```

### Missing Images
- Check that `image_root` path is correct
- Verify images exist at: `{image_root}/{access_no}/{sop_uid}.jpg`
- Failed copies are logged but don't stop the process

### YOLO Can't Find Dataset
- Ensure output_root contains train/, val/, or test/ subdirectories
- Each subdirectory should contain class_0/, class_1/, etc.
- Point YOLO to the root directory, not a subdirectory

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| df | DataFrame | Required | DataFrame with image metadata |
| image_root | str | Required | Source images directory |
| output_root | str | Required | Target YOLO dataset directory |
| class_column | str | 'bom' | Column name for class labels |
| df_name | str | 'Dataset' | Name for logging |
| split_name | str | 'train' | Split name ('train', 'val', 'test') |
| val_split | float | 0.2 | Validation split ratio (0.0-1.0) |

## Tips

- Use `val_split=0.2` for automatic 80/20 train/val split
- Use `val_split=0.0` when you don't want to split the data
- The split is stratified to maintain class balance
- Images are copied (not moved) to preserve originals
- Use unique `sop_uid` values to avoid filename collisions
