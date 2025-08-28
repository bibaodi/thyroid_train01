"""
Auto Annotate Tool

This script uses a CNN model (YOLOv11) to automatically generate annotations for images
and saves the results in LabelMe JSON format.

Usage:
    python auto_annotate.py --model_file <model_file> --model_type <segmentation|detection> \
                           --input_folder <input_folder> --output_folder <output_folder>

Author: eton
Version: 0.1
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
    print("WARNING: ultralytics package not found. YOLO models will not be available.")


def validate_args(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if all arguments are valid, False otherwise
    """
    logger = get_logger()
    
    # Check if model file exists
    if not os.path.exists(args.model_file):
        logger.error(f"Model file does not exist: {args.model_file}")
        return False
    
    # Check if model type is valid
    if args.model_type not in ["segmentation", "detection"]:
        logger.error(f"Invalid model type: {args.model_type}. Must be 'segmentation' or 'detection'.")
        return False
    
    # Check if input folder exists
    if not os.path.exists(args.input_folder):
        logger.error(f"Input folder does not exist: {args.input_folder}")
        return False
    
    # Check if input folder is actually a directory
    if not os.path.isdir(args.input_folder):
        logger.error(f"Input path is not a directory: {args.input_folder}")
        return False
    
    # Create output folder if it doesn't exist
    try:
        os.makedirs(args.output_folder, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output folder {args.output_folder}: {e}")
        return False
    
    return True


def predict_with_model(model_file: str, model_type: str, image_path: Path) -> List[Dict[str, Any]]:
    """
    Use the model to predict annotations for an image.
    
    Args:
        model_file: Path to the model file
        model_type: Type of model ('segmentation' or 'detection')
        image_path: Path to the image file
        
    Returns:
        List of predictions in a standardized format
    """
    logger = get_logger()
    
    if not YOLO_AVAILABLE:
        logger.error("YOLO is not available. Cannot perform prediction.")
        return []
    
    try:
        # Load the model
        model = YOLO(model_file)
        
        # Perform prediction
        results = model(str(image_path))
        
        # Process results
        predictions = []
        for result in results:
            # Get boxes or masks depending on model type
            if model_type == "detection" and hasattr(result, 'boxes'):
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
            elif model_type == "segmentation" and hasattr(result, 'masks'):
                masks = result.masks
                boxes = result.boxes
                
                for i, mask in enumerate(masks):
                    # Extract mask
                    mask_points = mask.xy[0]  # xy format
                    
                    # Get corresponding class and confidence
                    class_id = int(boxes.cls[i].cpu().numpy()) if boxes is not None else 0
                    confidence = float(boxes.conf[i].cpu().numpy()) if boxes is not None else 0.0
                    
                    # Convert to list of points
                    polygon = [[float(point[0]), float(point[1])] for point in mask_points]
                    
                    predictions.append({
                        "class_id": class_id,
                        "confidence": confidence,
                        "polygon": polygon
                    })
        
        logger.info(f"Predicted {len(predictions)} objects in {image_path.name}")
        return predictions
        
    except Exception as e:
        logger.error(f"Error during prediction for {image_path.name}: {e}")
        return []


def convert_to_labelme_format(predictions: List[Dict[str, Any]], image_path: Path) -> Dict[str, Any]:
    """
    Convert model predictions to LabelMe JSON format.
    
    Args:
        predictions: List of predictions from the model
        image_path: Path to the original image file
        
    Returns:
        Dictionary in LabelMe JSON format
    """
    logger = get_logger()
    
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
            
            # Set label (using class_id for now, could be mapped to class names)
            shape["label"] = f"class_{pred['class_id']}"
            
            # Set points
            shape["points"] = pred["polygon"]
            
            # Add to shapes
            labelme_data["shapes"].append(shape)
        
        logger.info(f"Converted {len(predictions)} predictions to LabelMe format for {image_path.name}")
        return labelme_data
        
    except Exception as e:
        logger.error(f"Error converting predictions to LabelMe format for {image_path.name}: {e}")
        return {}


def save_labelme_json(labelme_data: Dict[str, Any], output_path: Path) -> bool:
    """
    Save LabelMe JSON data to a file.
    
    Args:
        labelme_data: Dictionary in LabelMe JSON format
        output_path: Path where the JSON file should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = get_logger()
    
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved LabelMe JSON to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving LabelMe JSON to {output_path}: {e}")
        return False


def process_images(model_file: str, model_type: str, input_folder: Path, output_folder: Path) -> bool:
    """
    Process all images in the input folder using the model and save results in LabelMe format.
    
    Args:
        model_file: Path to the model file
        model_type: Type of model ('segmentation' or 'detection')
        input_folder: Path to the folder containing input images
        output_folder: Path to the folder where results should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = get_logger()
    
    try:
        # Get list of image files
        image_files = getImageFilesBySuffixes(input_folder)
        
        if not image_files:
            logger.warning(f"No image files found in {input_folder}")
            return False
        
        logger.info(f"Found {len(image_files)} image files in {input_folder}")
        
        # Process each image
        for image_path in image_files:
            logger.info(f"Processing {image_path.name}...")
            
            # Use model to predict
            predictions = predict_with_model(model_file, model_type, image_path)
            
            # Convert predictions to LabelMe format
            labelme_data = convert_to_labelme_format(predictions, image_path)
            
            # Calculate relative path from input folder
            relative_path = image_path.relative_to(input_folder)
            
            # Create output path with same relative structure
            output_path = output_folder / relative_path.with_suffix('.json')
            
            # Save LabelMe JSON
            save_labelme_json(labelme_data, output_path)
        
        logger.info(f"Processed {len(image_files)} images")
        return True
        
    except Exception as e:
        logger.error(f"Error processing images in {input_folder}: {e}")
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
    
    args = parser.parse_args()
    
    # Convert to Path objects
    model_file = Path(args.model_file)
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    # Validate arguments
    if not validate_args(args):
        logger.error("Argument validation failed. Exiting.")
        return 1
    
    # Process images
    logger.info("Starting auto annotation process...")
    success = process_images(str(model_file), args.model_type, input_folder, output_folder)
    
    if success:
        logger.info("Auto annotation process completed successfully.")
        return 0
    else:
        logger.error("Auto annotation process failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())