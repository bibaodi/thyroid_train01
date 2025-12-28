# Complete Filter System Usage Guide

## Overview

The complete filtering system includes orientation detection, thyroid nodule detection, and **nodule size filtering** using YOLO models. This allows you to automatically filter cases based on:
- Image orientations (transverse/longitudinal)
- Presence of thyroid nodules
- **Nodule size constraints (percentage of image size)**
- Minimum number of images
- File extensions and naming patterns

## Enhanced ThyroidNoduleDetector with Size Information

The ThyroidNoduleDetector now includes **size information** for each detected nodule:

```python
from srcs.lib.utils.thyroid_nodule_detector import ThyroidNoduleDetector

# Initialize with YOLO detection model
detector = ThyroidNoduleDetector('/path/to/nodule_model.pt', confidence_threshold=0.5)

# Detect nodules with size information
results = detector.detect_nodules(image_files)

# Enhanced results format with size info:
# [
#     {
#         'image_name': 'image1.png',
#         'image_size': {'width': 800, 'height': 600},
#         'has_nodules': True,
#         'nodule_count': 2,
#         'detections': [
#             {
#                 'bbox': [x1, y1, x2, y2],
#                 'confidence': 0.85,
#                 'class_name': 'thyroid_nodule',
#                 'size_info': {
#                     'width_pixels': 120.0,
#                     'height_pixels': 80.0,
#                     'width_percent': 15.0,    # 15% of image width
#                     'height_percent': 13.3,   # 13.3% of image height
#                     'area_percent': 2.0       # 2% of image area
#                 }
#             }
#         ]
#     }
# ]
```

## Complete Filter Configuration

```python
from srcs.predict_and_human_confirm import Predictor

# Configuration with all checks including nodule size filtering
filter_config = {
    "min_images_check": {
        "enabled": True,
        "min_count": 2
    },
    "orientation_check": {
        "enabled": True,
        "require_both_orientations": True
    },
    "thyroid_nodule_check": {
        "enabled": True,
        "require_nodules": True,
        "min_nodule_count": 1,
        "confidence_threshold": 0.5
    },
    "nodule_size_check": {
        "enabled": True,
        "size_ranges": [(0.1, 0.1), (0.5, 0.5)],  # 10% to 50% of image size
        "check_all_nodules": True,  # All nodules must meet size criteria
        "confidence_threshold": 0.5
    }
}

# Initialize predictor with all models
predictor = Predictor(
    model_path='/path/to/classification_model.pth',
    spreadsheet_file='/path/to/ground_truth.csv',
    filter_config=filter_config,
    orientation_model_path='/path/to/orientation_model.pt',
    thyroid_nodule_model_path='/path/to/nodule_detection_model.pt'
)

# Process cases - all filters applied automatically
predictor.process_root_folder('/path/to/cases', '/path/to/output')
```

## Nodule Size Filtering Examples

### Size Range Format
```python
size_ranges = [(min_width_percent, min_height_percent), (max_width_percent, max_height_percent)]
```

### Examples:

```python
# Example 1: Strict size filtering (all nodules must be 10-30% of image size)
filter_config = {
    "nodule_size_check": {
        "enabled": True,
        "size_ranges": [(0.1, 0.1), (0.3, 0.3)],  # 10% to 30% of image size
        "check_all_nodules": True,  # All nodules must meet criteria
        "confidence_threshold": 0.7
    }
}

# Example 2: Permissive size filtering (at least one nodule must be valid)
filter_config = {
    "nodule_size_check": {
        "enabled": True,
        "size_ranges": [(0.05, 0.05), (0.8, 0.8)],  # 5% to 80% of image size
        "check_all_nodules": False,  # At least one nodule must meet criteria
        "confidence_threshold": 0.3
    }
}

# Example 3: Asymmetric size requirements
filter_config = {
    "nodule_size_check": {
        "enabled": True,
        "size_ranges": [(0.15, 0.12), (0.4, 0.35)],  # Width: 15-40%, Height: 12-35%
        "check_all_nodules": True,
        "confidence_threshold": 0.6
    }
}
```

## Filter Behavior with Size Checking

### Example Output:
```
✅ Case 'case_001' passed filter checks, processing...
⚠️  Case 'case_002' failed filter checks:
   - orientation_check: Case must have both transverse and longitudinal images
   - thyroid_nodule_check: Case must have at least one image with thyroid nodules
   - nodule_size_check: Found 2/3 nodules with invalid sizes

⚠️  Case 'case_003' failed filter checks:
   - nodule_size_check: Found 1/1 nodules with invalid sizes; Size range: 10.0%-50.0% width, 10.0%-50.0% height
```

### Detailed Results:
```
Filter Results:
- min_images_check: ✅ Found 5 images, required minimum: 2
- orientation_check: ✅ Case has both orientations ✓
  Orientations: T=2, L=3, U=0
- thyroid_nodule_check: ✅ Case has images with thyroid nodules ✓
  Nodules: 4/5 images with nodules, 8 total nodules detected
- nodule_size_check: ✅ All 8 nodules have valid sizes ✓
  Size range: 10.0%-50.0% width, 10.0%-50.0% height
```

## Real Model Paths (Example)

```python
# Using actual model paths from your system
orientation_model = '/home/eton/00-src/task4efficientNet/models/model_Classify2ThyGlandPos_v01.250511/size224/model_Classify2GlandPos_v01S224.pt'
nodule_model = '/home/eton/00-src/task4efficientNet/models/model_detectThyroidNodules_v03.250309/model_detectThyroidNodules_v03.250309.pt'

predictor = Predictor(
    model_path='/path/to/classification_model.pth',
    spreadsheet_file='/path/to/data.csv',
    filter_config=filter_config,
    orientation_model_path=orientation_model,
    thyroid_nodule_model_path=nodule_model
)
```

## Available Filter Nodes

1. **min_images_check**: Minimum number of images required
2. **max_images_check**: Maximum number of images allowed
3. **file_extension_check**: Valid file extensions
4. **case_name_pattern_check**: Case name regex pattern matching
5. **orientation_check**: Image orientation requirements (transverse/longitudinal)
6. **thyroid_nodule_check**: Thyroid nodule presence requirements
7. **nodule_size_check**: Nodule size constraints (NEW!)

## Key Features

- **Size Percentage Calculation**: Nodule dimensions as percentage of image size
- **Flexible Size Constraints**: Configure min/max width and height percentages
- **All vs Any Logic**: Choose whether all nodules or just one must meet size criteria
- **Detailed Size Statistics**: Average, min, max size information in summaries
- **Size Violation Reporting**: Detailed information about which nodules fail size checks
- **m_ Prefix**: All class member variables use m_ prefix for consistency
- **YOLO Integration**: Both models use ultralytics YOLO framework
- **Graceful Degradation**: Works without models (filters disabled with warnings)

## Dependencies

```bash
pip install ultralytics  # Required for YOLO models
```

## Performance Notes

- **Enhanced Detection**: Size calculation adds minimal overhead to detection
- **Memory Efficient**: Size information stored efficiently with detection results
- **Detailed Logging**: Comprehensive size violation reporting for debugging
- **Early Filtering**: Cases filtered before expensive processing, saving time
