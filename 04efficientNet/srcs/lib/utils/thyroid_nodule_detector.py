"""
Thyroid Nodule Detector
Detects thyroid nodules in ultrasound images using YOLO-11n model
"""
import os
import torch
from PIL import Image
from typing import List, Dict, Union, Any
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️  Ultralytics not available. Please install: pip install ultralytics")


class ThyroidNoduleDetector:
    """
    Detects thyroid nodules in ultrasound images using YOLO-11n model
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = None):
        """
        Initialize the thyroid nodule detector.

        Args:
            model_path: Path to the trained YOLO model
            confidence_threshold: Minimum confidence for detections (default: 0.5)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics is required for thyroid nodule detection. Install with: pip install ultralytics")

        self.m_model_path = model_path
        self.m_confidence_threshold = confidence_threshold
        self.m_device = self._get_device(device)
        self.m_model = None

        # Initialize model
        self._load_model()

    def _get_device(self, device: str = None) -> str:
        """Get the appropriate device for inference."""
        if device:
            return device

        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def _load_model(self):
        """Load the YOLO thyroid nodule detection model."""
        if not os.path.exists(self.m_model_path):
            raise FileNotFoundError(f"Model file not found: {self.m_model_path}")

        try:
            # Load YOLO model
            self.m_model = YOLO(self.m_model_path)

            # Set device
            if hasattr(self.m_model, 'to'):
                self.m_model.to(self.m_device)

            print(f"✅ Thyroid nodule detection model loaded from: {self.m_model_path}")
            print(f"   Device: {self.m_device}")
            print(f"   Confidence threshold: {self.m_confidence_threshold}")

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def _detect_nodules_in_image(self, image_path: str) -> Dict[str, Any]:
        """
        Detect thyroid nodules in a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with detection results including size information
        """
        try:
            # Get original image dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            # Run inference
            results = self.m_model(image_path, conf=self.m_confidence_threshold, device=self.m_device, verbose=False)

            # Extract detection information
            detections = []
            has_nodules = False

            if results and len(results) > 0:
                result = results[0]  # First (and only) image result

                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        # Get box coordinates and confidence
                        box = boxes.xyxy[i].cpu().numpy() if hasattr(boxes, 'xyxy') else None
                        conf = boxes.conf[i].cpu().numpy() if hasattr(boxes, 'conf') else 0.0
                        cls = boxes.cls[i].cpu().numpy() if hasattr(boxes, 'cls') else 0

                        if conf >= self.m_confidence_threshold and box is not None:
                            has_nodules = True

                            # Calculate nodule dimensions
                            x1, y1, x2, y2 = box
                            nodule_width = x2 - x1
                            nodule_height = y2 - y1

                            # Calculate size as percentage of image
                            width_percent = (nodule_width / img_width) * 100
                            height_percent = (nodule_height / img_height) * 100

                            # Calculate area percentage
                            nodule_area = nodule_width * nodule_height
                            image_area = img_width * img_height
                            area_percent = (nodule_area / image_area) * 100

                            detections.append({
                                'bbox': box.tolist(),
                                'confidence': float(conf),
                                'class': int(cls),
                                'class_name': 'thyroid_nodule',
                                'size_info': {
                                    'width_pixels': float(nodule_width),
                                    'height_pixels': float(nodule_height),
                                    'width_percent': float(width_percent),
                                    'height_percent': float(height_percent),
                                    'area_percent': float(area_percent)
                                }
                            })

            return {
                'image_path': image_path,
                'image_size': {'width': img_width, 'height': img_height},
                'has_nodules': has_nodules,
                'nodule_count': len(detections),
                'detections': detections,
                'confidence_threshold': self.m_confidence_threshold
            }

        except Exception as e:
            print(f"⚠️  Error detecting nodules in {image_path}: {e}")
            return {
                'image_path': image_path,
                'image_size': {'width': 0, 'height': 0},
                'has_nodules': False,
                'nodule_count': 0,
                'detections': [],
                'error': str(e)
            }

    def detect_nodules(self, images: Union[List[str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Detect thyroid nodules in a list of images.

        Args:
            images: Either:
                - List of image file paths
                - List of dicts with 'image_name' key and image data

        Returns:
            List of dicts with nodule detection results
        """
        results = []

        for item in images:
            try:
                if isinstance(item, str):
                    # Item is a file path
                    image_path = item
                    image_name = os.path.basename(image_path)

                elif isinstance(item, dict) and 'image_name' in item:
                    # Item is a dict with image data
                    image_name = item['image_name']

                    if 'image_path' in item:
                        # Use provided path
                        image_path = item['image_path']
                    else:
                        # Skip if no path provided for dict input
                        print(f"⚠️  No image_path provided for {image_name}")
                        results.append({
                            'image_name': image_name,
                            'has_nodules': False,
                            'nodule_count': 0,
                            'detections': [],
                            'error': 'No image path provided'
                        })
                        continue
                else:
                    raise ValueError(f"Unsupported input type: {type(item)}")

                # Detect nodules
                detection_result = self._detect_nodules_in_image(image_path)
                detection_result['image_name'] = image_name

                results.append(detection_result)

            except Exception as e:
                print(f"⚠️  Error processing image {item}: {e}")
                results.append({
                    'image_name': getattr(item, 'image_name', str(item)),
                    'has_nodules': False,
                    'nodule_count': 0,
                    'detections': [],
                    'error': str(e)
                })

        return results

    def get_nodule_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of nodule detection results.

        Args:
            results: Results from detect_nodules()

        Returns:
            Dictionary with nodule detection statistics including size information
        """
        total_images = len(results)
        images_with_nodules = sum(1 for r in results if r.get('has_nodules', False))
        images_without_nodules = total_images - images_with_nodules
        total_nodules = sum(r.get('nodule_count', 0) for r in results)

        # Calculate average nodules per image (for images with nodules)
        avg_nodules_per_image = 0
        if images_with_nodules > 0:
            avg_nodules_per_image = total_nodules / images_with_nodules

        # Find images with errors
        error_count = sum(1 for r in results if 'error' in r)

        # Collect size statistics
        all_nodules = []
        for result in results:
            for detection in result.get('detections', []):
                if 'size_info' in detection:
                    all_nodules.append(detection['size_info'])

        # Calculate size statistics
        size_stats = {}
        if all_nodules:
            width_percents = [n['width_percent'] for n in all_nodules]
            height_percents = [n['height_percent'] for n in all_nodules]
            area_percents = [n['area_percent'] for n in all_nodules]

            size_stats = {
                'avg_width_percent': round(sum(width_percents) / len(width_percents), 2),
                'avg_height_percent': round(sum(height_percents) / len(height_percents), 2),
                'avg_area_percent': round(sum(area_percents) / len(area_percents), 2),
                'min_width_percent': round(min(width_percents), 2),
                'max_width_percent': round(max(width_percents), 2),
                'min_height_percent': round(min(height_percents), 2),
                'max_height_percent': round(max(height_percents), 2),
                'min_area_percent': round(min(area_percents), 2),
                'max_area_percent': round(max(area_percents), 2)
            }

        return {
            'total_images': total_images,
            'images_with_nodules': images_with_nodules,
            'images_without_nodules': images_without_nodules,
            'total_nodules_detected': total_nodules,
            'avg_nodules_per_image_with_nodules': round(avg_nodules_per_image, 2),
            'error_count': error_count,
            'has_any_nodules': images_with_nodules > 0,
            'detection_rate': round(images_with_nodules / total_images * 100, 2) if total_images > 0 else 0,
            'size_statistics': size_stats
        }

    def set_confidence_threshold(self, threshold: float):
        """Update the confidence threshold for detections."""
        self.m_confidence_threshold = max(0.0, min(1.0, threshold))
        print(f"Confidence threshold updated to: {self.m_confidence_threshold}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_path': self.m_model_path,
            'device': self.m_device,
            'confidence_threshold': self.m_confidence_threshold,
            'model_loaded': self.m_model is not None
        }

        if self.m_model is not None:
            try:
                # Try to get model info from YOLO
                if hasattr(self.m_model, 'info'):
                    self.m_model.info()
                info['model_type'] = 'YOLO'
            except:
                pass

        return info
