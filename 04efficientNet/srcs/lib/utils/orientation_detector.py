"""
Image Orientation Detector
Detects whether ultrasound images are transverse or longitudinal using YOLO model
"""
import os
import torch
from PIL import Image
from typing import List, Dict, Union, Any
import numpy as np

from .enums import ImageOrientation

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️  Ultralytics not available. Please install: pip install ultralytics")


class OrientationDetector:
    """
    Detects image orientation (transverse vs longitudinal) using a YOLO classification model
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the orientation detector.

        Args:
            model_path: Path to the trained YOLO classification model
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
        """
        if not ULTRALYTICS_AVAILABLE:
            print("⚠️  Ultralytics not available. Using placeholder model for demonstration")

        self.m_model_path = model_path
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
        """Load the YOLO orientation classification model."""
        if not os.path.exists(self.m_model_path):
            print(f"⚠️  Model file not found: {self.m_model_path}")
            print("   Using placeholder model for demonstration")
            return

        if not ULTRALYTICS_AVAILABLE:
            print("⚠️  Ultralytics not available. Using placeholder model")
            return

        try:
            # Load YOLO model
            self.m_model = YOLO(self.m_model_path)

            # Set device
            if hasattr(self.m_model, 'to'):
                self.m_model.to(self.m_device)

            print(f"✅ Orientation model loaded from: {self.m_model_path}")
            print(f"   Device: {self.m_device}")

        except Exception as e:
            print(f"⚠️  Could not load YOLO model: {e}")
            print("   Using placeholder model for demonstration")
            self.m_model = None

    def _predict_orientation(self, image_path: str) -> ImageOrientation:
        """
        Predict orientation for a single image.

        Args:
            image_path: Path to the image file

        Returns:
            ImageOrientation enum value
        """
        if self.m_model is None:
            # Return random orientation for demonstration
            import random
            return random.choice([ImageOrientation.TRANSVERSE, ImageOrientation.LONGITUDINAL])

        try:
            # Run YOLO classification
            results = self.m_model(image_path, device=self.m_device, verbose=False)

            if results and len(results) > 0:
                result = results[0]  # First (and only) image result

                # Get the predicted class
                if hasattr(result, 'probs') and result.probs is not None:
                    # Classification result
                    predicted_class = result.probs.top1
                    confidence = result.probs.top1conf.item()

                    # Map class index to orientation
                    # Assuming: 0 = transverse, 1 = longitudinal
                    if predicted_class == 0:
                        return ImageOrientation.TRANSVERSE
                    elif predicted_class == 1:
                        return ImageOrientation.LONGITUDINAL
                    else:
                        return ImageOrientation.UNKNOWN
                else:
                    return ImageOrientation.UNKNOWN
            else:
                return ImageOrientation.UNKNOWN

        except Exception as e:
            print(f"⚠️  Error predicting orientation for {image_path}: {e}")
            return ImageOrientation.UNKNOWN

    def detect_orientations(self,
                          images: Union[List[str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Detect orientations for a list of images.

        Args:
            images: Either:
                - List of image file paths
                - List of dicts with 'image_name' key and image data

        Returns:
            List of dicts with 'image_name' and 'orientation' keys
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
                        # Load from path
                        image_path = item['image_path']
                    else:
                        # Skip if no path provided for dict input
                        print(f"⚠️  No image_path provided for {image_name}")
                        results.append({
                            'image_name': image_name,
                            'orientation': ImageOrientation.UNKNOWN
                        })
                        continue
                else:
                    raise ValueError(f"Unsupported input type: {type(item)}")

                # Predict orientation
                orientation = self._predict_orientation(image_path)

                results.append({
                    'image_name': image_name,
                    'orientation': orientation
                })

            except Exception as e:
                print(f"⚠️  Error processing image {item}: {e}")
                results.append({
                    'image_name': getattr(item, 'image_name', str(item)),
                    'orientation': ImageOrientation.UNKNOWN
                })

        return results

    def get_orientation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics of orientation detection results.

        Args:
            results: Results from detect_orientations()

        Returns:
            Dictionary with orientation counts and statistics
        """
        orientation_counts = {
            ImageOrientation.TRANSVERSE: 0,
            ImageOrientation.LONGITUDINAL: 0,
            ImageOrientation.UNKNOWN: 0
        }

        for result in results:
            orientation = result['orientation']
            orientation_counts[orientation] += 1

        total_images = len(results)

        return {
            'total_images': total_images,
            'transverse_count': orientation_counts[ImageOrientation.TRANSVERSE],
            'longitudinal_count': orientation_counts[ImageOrientation.LONGITUDINAL],
            'unknown_count': orientation_counts[ImageOrientation.UNKNOWN],
            'has_transverse': orientation_counts[ImageOrientation.TRANSVERSE] > 0,
            'has_longitudinal': orientation_counts[ImageOrientation.LONGITUDINAL] > 0,
            'has_both_orientations': (
                orientation_counts[ImageOrientation.TRANSVERSE] > 0 and
                orientation_counts[ImageOrientation.LONGITUDINAL] > 0
            )
        }
