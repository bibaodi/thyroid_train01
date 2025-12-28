# Enhanced Filter System Usage Guide

## Overview

The enhanced filtering system includes both orientation detection (transverse vs longitudinal) and thyroid nodule detection using YOLO models. This allows you to automatically filter cases based on:
- Image orientations (transverse/longitudinal)
- Presence of thyroid nodules
- Minimum number of images
- File extensions and naming patterns

## Components

### 1. ImageOrientation Enum (`srcs/lib/utils/enums.py`)
```python
from srcs.lib.utils.enums import ImageOrientation

# Available orientations:
ImageOrientation.TRANSVERSE    # "transverse"
ImageOrientation.LONGITUDINAL  # "longitudinal"
ImageOrientation.UNKNOWN       # "unknown"
```

### 2. OrientationDetector (`srcs/lib/utils/orientation_detector.py`)
Now uses YOLO classification model:
```python
from srcs.lib.utils.orientation_detector import OrientationDetector

# Initialize with YOLO classification model
detector = OrientationDetector('/path/to/orientation_model.pt')

# Detect orientations for a list of image files
image_files = ['/path/to/image1.png', '/path/to/image2.png']
results = detector.detect_orientations(image_files)
```

### 3. ThyroidNoduleDetector (`srcs/lib/utils/thyroid_nodule_detector.py`)
Uses YOLO-11n detection model:
```python
from srcs.lib.utils.thyroid_nodule_detector import ThyroidNoduleDetector

# Initialize with YOLO detection model
detector = ThyroidNoduleDetector('/path/to/nodule_model.pt', confidence_threshold=0.5)

# Detect nodules in images
results = detector.detect_nodules(image_files)

# Results format:
# [
#     {
#         'image_name': 'image1.png',
#         'has_nodules': True,
#         'nodule_count': 2,
#         'detections': [...]
#     }
# ]
```

## Usage Examples

### Complete Filter Configuration

```python
from srcs.predict_and_human_confirm import Predictor

# Configuration with all checks enabled
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
    }
}

# Initialize predictor with both models
predictor = Predictor(
    model_path='/path/to/classification_model.pth',
    spreadsheet_file='/path/to/ground_truth.csv',
    filter_config=filter_config,
    orientation_model_path='/path/to/orientation_model.pt',        # YOLO classification
    thyroid_nodule_model_path='/path/to/nodule_detection_model.pt' # YOLO detection
)

# Process cases - all filters applied automatically
predictor.process_root_folder('/path/to/cases', '/path/to/output')
```

### Real Model Paths (Example)

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

## Filter Behavior

### Example Output:
```
✅ Case 'case_001' passed filter checks, processing...
⚠️  Case 'case_002' failed filter checks:
   - orientation_check: Case must have both transverse and longitudinal images
   - thyroid_nodule_check: Case must have at least one image with thyroid nodules
```

### Detailed Results:
```
Filter Results:
- min_images_check: ✅ Found 5 images, required minimum: 2
- orientation_check: ✅ Case has both orientations ✓
  Orientations: T=2, L=3, U=0
- thyroid_nodule_check: ✅ Case has images with thyroid nodules ✓
  Nodules: 3/5 images with nodules, 7 total nodules detected
```

## Model Requirements

### Orientation Model (YOLO Classification):
- **Format**: YOLO classification model (.pt file)
- **Classes**: 2 classes [transverse, longitudinal]
- **Example**: `model_Classify2GlandPos_v01S224.pt`

### Thyroid Nodule Model (YOLO Detection):
- **Format**: YOLO-11n detection model (.pt file)
- **Task**: Object detection for thyroid nodules
- **Example**: `model_detectThyroidNodules_v03.250309.pt`

## Dependencies

```bash
pip install ultralytics  # Required for YOLO models
```

## Key Features

- **m_ Prefix**: All class member variables use m_ prefix for consistency
- **YOLO Integration**: Both models use ultralytics YOLO framework
- **Graceful Degradation**: Works without models (filters disabled with warnings)
- **Detailed Logging**: Comprehensive feedback on filter results
- **Resource Efficient**: Early filtering saves computation time
- **Configurable**: Easy to enable/disable individual checks
