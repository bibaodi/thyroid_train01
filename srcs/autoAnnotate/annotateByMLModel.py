"""
Auto Annotate Tool

This script uses a CNN model (YOLOv11) to automatically generate annotations for images
and saves the results in LabelMe JSON format.

Usage:
    python auto_annotate.py --model_file <model_file> --model_type <segmentation|detection> \
                           --input_folder <input_folder> --output_folder <output_folder>

Author: eton
Version: 0.2
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project utilities
from utils.glog import initLogger, get_logger
from annotationFormatConverter.LabelmeJson import getOneTargetObj, getOneShapeObj, getImageFilesBySuffixes

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    get_logger().fatal("WARNING: ultralytics package not found. YOLO models will not be available.")


class AutoAnnotator:
    """Class to handle automatic annotation of images using YOLO models."""
    
    def __init__(self, model_file: str, model_type: str, input_folder: Path, 
                 output_folder: Path, label_name: str = None):
        """
        Initialize the AutoAnnotator.
        
        Args:
            model_file: Path to the model file
            model_type: Type of model ('segmentation' or 'detection')
            input_folder: Path to the folder containing input images
            output_folder: Path to the folder where results should be saved
            label_name: Label name to use for all annotations (optional)
        """
        self.m_model_file = model_file
        self.m_model_type = model_type
        self.m_input_folder = input_folder
        self.m_output_folder = output_folder
        self.m_label_name = label_name
        self.m_logger = get_logger()
    
    def _validate_args(self) -> bool:
        """
        Validate initialization arguments.
        
        Returns:
            bool: True if all arguments are valid, False otherwise
        """
        # Check if model file exists
        if not os.path.exists(self.m_model_file):
            self.m_logger.error(f"Model file does not exist: {self.m_model_file}")
            return False
        
        # Check if model type is valid
        if self.m_model_type not in ["segmentation", "detection"]:
            self.m_logger.error(f"Invalid model type: {self.m_model_type}. Must be 'segmentation' or 'detection'.")
            return False
        
        # Check if input folder exists
        if not os.path.exists(self.m_input_folder):
            self.m_logger.error(f"Input folder does not exist: {self.m_input_folder}")
            return False
        
        # Check if input folder is actually a directory
        if not os.path.isdir(self.m_input_folder):
            self.m_logger.error(f"Input path is not a directory: {self.m_input_folder}")
            return False
        
        # Create output folder if it doesn't exist
        try:
            os.makedirs(self.m_output_folder, exist_ok=True)
        except Exception as e:
            self.m_logger.error(f"Failed to create output folder {self.m_output_folder}: {e}")
            return False
        
        return True
    
    def _predict_with_model(self, image_path: Path) -> tuple[List[Dict[str, Any]], Dict[int, str]]:
        """
        Use the model to predict annotations for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predictions, class_names)
        """
        if not YOLO_AVAILABLE:
            self.m_logger.error("YOLO is not available. Cannot perform prediction.")
            return [], {}
        
        try:
            # Load the model
            model = YOLO(self.m_model_file)
            
            # Get class names if available
            class_names = {}
            if hasattr(model, 'names'):
                class_names = model.names
            
            # Perform prediction
            results = model(str(image_path))
            
            # Process results
            predictions = []
            for result in results:
                # Get boxes or masks depending on model type
                if self.m_model_type == "detection" and hasattr(result, 'boxes'):
                    boxes = result.boxes
                    for box in boxes:
                        # Extract box coordinates
                        xyxy = box.xyxy[0].cpu().numpy()  # xyxy format
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Convert to polygon format (rectangle)
                        x1, y1, x2, y2 = xyxy
                        polygon = [
                            [float(x1), float(y1)],
                            [float(x2), float(y1)],
                            [float(x2), float(y2)],
                            [float(x1), float(y2)]
                        ]
                        
                        predictions.append({
                            "class_id": class_id,
                            "confidence": confidence,
                            "polygon": polygon
                        })
                elif self.m_model_type == "segmentation" and hasattr(result, 'masks'):
                    masks = result.masks
                    boxes = result.boxes
                    
                    for i, mask in enumerate(masks):
                        # Extract mask points - this gives the actual contour points
                        mask_points = mask.xy[0]  # xy format
                        
                        # Get corresponding class and confidence
                        class_id = int(boxes.cls[i].cpu().numpy()) if boxes is not None else 0
                        confidence = float(boxes.conf[i].cpu().numpy()) if boxes is not None else 0.0
                        
                        # Convert to list of points
                        # Ensure we have enough points for a proper polygon
                        if len(mask_points) >= 3:  # Need at least 3 points for a polygon
                            polygon = [[float(point[0]), float(point[1])] for point in mask_points]
                            
                            predictions.append({
                                "class_id": class_id,
                                "confidence": confidence,
                                "polygon": polygon
                            })
                        else:
                            self.m_logger.warning(f"Skipping mask with insufficient points: {len(mask_points)}")      
            self.m_logger.info(f"Predicted {len(predictions)} objects in {image_path.name}")
            return predictions, class_names
            
        except Exception as e:
            self.m_logger.error(f"Error during prediction for {image_path.name}: {e}")
            return [], {}
    
    def _convert_to_labelme_format(self, predictions: List[Dict[str, Any]], image_path: Path, 
                                  class_names: Dict[int, str]) -> Dict[str, Any]:
        """
        Convert model predictions to LabelMe JSON format.
        
        Args:
            predictions: List of predictions from the model
            image_path: Path to the original image file
            class_names: Dictionary mapping class IDs to class names from the model
            
        Returns:
            Dictionary in LabelMe JSON format
        """
        try:
            # Create a new LabelMe JSON object
            labelme_data = getOneTargetObj()
            
            # Clear existing shapes
            labelme_data["shapes"].clear()
            
            # Set image path
            labelme_data["imagePath"] = image_path.name
            
            # Set image data to None (will be loaded by LabelMe when needed)
            labelme_data["imageData"] = None
            
            # Add predictions as shapes
            for pred in predictions:
                shape = getOneShapeObj()
                
                # Set label based on priority:
                # 1. Use m_label_name if provided
                # 2. Use class name from model if available
                # 3. Use class_id as fallback
                if self.m_label_name:
                    shape["label"] = self.m_label_name
                elif class_names and pred['class_id'] in class_names:
                    shape["label"] = class_names[pred['class_id']]
                else:
                    shape["label"] = f"class_{pred['class_id']}"
                
                # Set points
                shape["points"] = pred["polygon"]
                
                # Add to shapes
                labelme_data["shapes"].append(shape)
            
            self.m_logger.info(f"Converted {len(predictions)} predictions to LabelMe format for {image_path.name}")
            return labelme_data
            
        except Exception as e:
            self.m_logger.error(f"Error converting predictions to LabelMe format for {image_path.name}: {e}")
            return {}
    
    def _save_labelme_json(self, labelme_data: Dict[str, Any], output_path: Path) -> bool:
        """
        Save LabelMe JSON data to a file.
        
        Args:
            labelme_data: Dictionary in LabelMe JSON format
            output_path: Path where the JSON file should be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)
            
            self.m_logger.info(f"Saved LabelMe JSON to {output_path}")
            return True
            
        except Exception as e:
            self.m_logger.error(f"Error saving LabelMe JSON to {output_path}: {e}")
            return False
    
    def process_images(self) -> bool:
        """
        Process all images in the input folder using the model and save results in LabelMe format.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate arguments first
            if not self._validate_args():
                self.m_logger.error("Argument validation failed. Exiting.")
                return False
            
            # Get list of image files
            image_files = getImageFilesBySuffixes(self.m_input_folder)
            
            if not image_files:
                self.m_logger.warning(f"No image files found in {self.m_input_folder}")
                return False
            
            self.m_logger.info(f"Found {len(image_files)} image files in {self.m_input_folder}")
            
            # Process each image
            for image_path in image_files:
                self.m_logger.info(f"Processing {image_path.name}...")
                
                # Use model to predict
                predictions, class_names = self._predict_with_model(image_path)
                
                # Convert predictions to LabelMe format
                labelme_data = self._convert_to_labelme_format(predictions, image_path, class_names)
                
                # Calculate relative path from input folder
                relative_path = image_path.relative_to(self.m_input_folder)
                
                # Create output path with same relative structure
                output_path = self.m_output_folder / relative_path.with_suffix('.json')
                
                # Save LabelMe JSON
                self._save_labelme_json(labelme_data, output_path)
            
            self.m_logger.info(f"Processed {len(image_files)} images")
            return True
            
        except Exception as e:
            self.m_logger.error(f"Error processing images in {self.m_input_folder}: {e}")
            return False


def main():
    """
    Main function to parse arguments and run the auto annotation process.
    """
    # Initialize logger
    initLogger("auto_annotate")
    logger = get_logger()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Auto annotate images using a CNN model')
    parser.add_argument('--model_file', required=True, help='Path to the model file')
    parser.add_argument('--model_type', required=True, choices=['segmentation', 'detection'], 
                        help='Type of model (segmentation or detection)')
    parser.add_argument('--input_folder', required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_folder', required=True, help='Path to the folder where results should be saved')
    parser.add_argument('--label_name', default=None, help='Label name to use for all annotations (optional, uses model class names if available)')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    model_file = Path(args.model_file)
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    # Create annotator instance
    annotator = AutoAnnotator(str(model_file), args.model_type, input_folder, output_folder, args.label_name)
    
    # Process images
    logger.info("Starting auto annotation process...")
    success = annotator.process_images()
    
    if success:
        logger.info("Auto annotation process completed successfully.")
        return 0
    else:
        logger.error("Auto annotation process failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())