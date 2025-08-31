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
import signal
from pathlib import Path
from typing import List, Dict, Any
import threading, queue

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project utilities
from utils.glog import initLogger, get_logger
from annotationFormatConverter.LabelmeJson import getOneTargetObj, getOneShapeObj, getImageFilesBySuffixes, process_polygon_points

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    get_logger().fatal("WARNING: ultralytics package not found. YOLO models will not be available.")

def calculate_dice_similarity(points1: List[List[float]], points2: List[List[float]]) -> float:
    """
    Calculate DICE similarity between two polygon shapes.
    
    Args:
        points1: First polygon points
        points2: Second polygon points
        
    Returns:
        float: DICE similarity coefficient (0.0 to 1.0)
    """
    if not points1 or not points2:
        return 0.0
    
    try:
        # Convert points to numpy arrays
        poly1 = np.array(points1, dtype=np.int32)
        poly2 = np.array(points2, dtype=np.int32)
        
        # Create blank masks
        # Determine the bounding box for both polygons
        all_points = np.concatenate([poly1, poly2])
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)
        
        # Create masks with some padding
        width = max_x - min_x + 20
        height = max_y - min_y + 20
        
        mask1 = np.zeros((height, width), dtype=np.uint8)
        mask2 = np.zeros((height, width), dtype=np.uint8)
        
        # Adjust points relative to the bounding box
        adjusted_poly1 = poly1 - [min_x - 10, min_y - 10]
        adjusted_poly2 = poly2 - [min_x - 10, min_y - 10]
        
        # Draw polygons on masks
        cv2.fillPoly(mask1, [adjusted_poly1], 1)
        cv2.fillPoly(mask2, [adjusted_poly2], 1)
        
        # Calculate DICE coefficient
        intersection = np.sum(mask1 * mask2)
        area1 = np.sum(mask1)
        area2 = np.sum(mask2)
        
        if area1 + area2 == 0:
            return 0.0
            
        dice = 2 * intersection / (area1 + area2)
        return dice
        
    except Exception as e:
        return 0.0
        
def isDuplicateShape(new_shape: Dict[str, Any], existing_shapes: List[Dict[str, Any]], 
                    dice_threshold: float = 0.8) -> bool:
    """
    Check if a new shape is a duplicate of any existing shape using DICE similarity.
    
    Args:
        new_shape: New shape to check
        existing_shapes: List of existing shapes
        dice_threshold: DICE similarity threshold for considering shapes as duplicates (default: 0.8)
        
    Returns:
        bool: True if duplicate found, False otherwise
    """
    new_points = new_shape.get("points", [])
    
    for existing_shape in existing_shapes:
        # Check if labels match
        if new_shape.get("label") != existing_shape.get("label"):
            continue
            
        existing_points = existing_shape.get("points", [])
        
        # Calculate DICE similarity
        dice_score = calculate_dice_similarity(new_points, existing_points)
        # If similarity exceeds threshold, consider it a duplicate
        if dice_score >= dice_threshold:
            return True
    return False

class AutoAnnotator:
    """Class to handle automatic annotation of images using YOLO models."""
    
    # Class variable to handle exit signals
    exit_event = threading.Event()
    
    def __init__(self, model_file: str, model_type: str, input_folder: Path, 
                 output_folder: Path, label_name: str = None, jobsCount: int = 1):
        """
        Initialize the AutoAnnotator.
        
        Args:
            model_file: Path to the model file
            model_type: Type of model ('segmentation' or 'detection')
            input_folder: Path to the folder containing input images
            output_folder: Path to the folder where results should be saved
            label_name: Label name to use for all annotations (optional)
            jobs: Number of parallel jobs for processing images
        """
        self.m_model_file = model_file
        self.m_model_type = model_type
        self.m_input_folder = input_folder
        self.m_output_folder = output_folder
        self.m_label_name = label_name
        self.m_jobsCount = jobsCount
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
    
    def _predict_detection(self, results) -> List[Dict[str, Any]]:
        """
        Process detection model results.
        
        Args:
            results: Model prediction results
            
        Returns:
            List of predictions
        """
        predictions = []
        for result in results:
            if hasattr(result, 'boxes'):
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
        return predictions
    
    def _predict_segmentation(self, results) -> List[Dict[str, Any]]:
        """
        Process segmentation model results.
        Args:
            results: Model prediction results
        Returns:
            List of predictions
        """
        predictions = []
        for result in results:
            if hasattr(result, 'masks'):
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
        return predictions
    

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
            self.m_logger.info(f"Loaded model {self.m_model_file} with classNames: {class_names}")
            
            # Perform prediction
            results = model.predict(str(image_path), verbose=False) # Add verbose=False here to suppress model predict messages)
            
            # Process results based on model type
            if self.m_model_type == "detection":
                predictions = self._predict_detection(results)
            elif self.m_model_type == "segmentation":
                predictions = self._predict_segmentation(results)
            else:
                self.m_logger.error(f"Invalid model type: {self.m_model_type}")
                return [], {}
            
            self.m_logger.info(f"Predicted {len(predictions)} objects in {image_path.name}")
            return predictions, class_names
            
        except Exception as e:
            self.m_logger.error(f"Error during prediction for {image_path.name}: {e}")
            return [], {}

    def _load_existing_lbmejson(self, json_path: Path) -> List[Dict[str, Any]]:
        """
        Load existing shapes from a JSON file if it exists.
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of existing shapes, or empty list if file doesn't exist or is invalid
        """
        if not json_path.exists():
            return []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                jsonObj = json.load(f)
                if isinstance(jsonObj, dict) and 'shapes' in jsonObj:
                    return jsonObj
        except Exception as e:
            self.m_logger.warning(f"Could not load existing JSON file {json_path}: {e}")
        
        return []
    
    def _get_labelName(self, pred_class_id:str, class_names: Dict[int, str]) -> str:
        shape_labelname='objLabelName'
        if self.m_label_name:
            shape_labelname = self.m_label_name
        elif class_names and pred_class_id in class_names:
            shape_labelname = class_names[pred_class_id]
        else:
            shape_labelname = f"class_{pred_class_id}"
            
        return shape_labelname
    
    def _convert_to_labelme_format(self, predictions: List[Dict[str, Any]], image_path: Path, 
                              class_names: Dict[int, str], output_path: Path) -> Dict[str, Any]:
        """
        Convert model predictions to LabelMe JSON format.
        
        Args:
            predictions: List of predictions from the model
            image_path: Path to the original image file
            class_names: Dictionary mapping class IDs to class names from the model
            output_path: Path where the JSON file should be saved
            
        Returns:
            Dictionary in LabelMe JSON format
        """
        try:
            # Create a new LabelMe JSON object
            labelme_data = getOneTargetObj()
            # Load existing shapes if file exists
            existingJsonObj = self._load_existing_lbmejson(output_path)
            
            # Set shapes to existing shapes if any
            if existingJsonObj:
                labelme_data = existingJsonObj
                existingShapes = existingJsonObj["shapes"]
                self.m_logger.info(f"Loaded {len(existingShapes)} existing shapes from {output_path}")
            else:
                self.m_logger.info(f"NO existing shapes from {output_path}, will create new one")
                # Clear existing shapes only if no existing file
                labelme_data["shapes"].clear()            
                # Set image path
                labelme_data["imagePath"] = image_path.name
                # Set image data to None (will be loaded by LabelMe when needed)
                labelme_data["imageData"] = None
            shapesInJson = labelme_data["shapes"]
            added_shapes = 0
            # Add predictions as shapes
            for pred in predictions:
                oneshape = getOneShapeObj()
                # Set label based on priority:
                # 1. Use m_label_name if provided
                # 2. Use class name from model if available
                # 3. Use class_id as fallback
                shape_labelname=self._get_labelName(pred['class_id'], class_names)
                oneshape["label"] = shape_labelname
                # Process points to ensure integer coordinates and no negative values
                oneshape["points"] = process_polygon_points(pred["polygon"])
                
                # check if shape already exist
                if isDuplicateShape(oneshape, shapesInJson, dice_threshold=0.8):
                    self.m_logger.debug(f"Skipped duplicate shape: {shape_labelname}")
                else:
                    shapesInJson.append(oneshape)
                    added_shapes += 1
            
            self.m_logger.info(f"Converted {added_shapes} new {shape_labelname} predictions to LabelMe format for {image_path.name} (skipped duplicates)")
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

    def process_images_inBatch(self, jobid:int, image_paths: List[Path]) -> bool:
        """
        Process a ibatch of images using the model and save results in LabelMe format.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for image_path in tqdm(image_paths, desc=f"Job{jobid}:Processing images", unit="image"):
                # Check if we should exit
                if AutoAnnotator.exit_event.is_set():
                    self.m_logger.info(f"Thread {jobid} received exit signal, terminating...")
                    return False
                
                self.m_logger.info(f"Processing {image_path.name}...")
                # Use model to predict
                predictions, class_names = self._predict_with_model(image_path)
                # Calculate relative path from input folder
                relative_path = image_path.relative_to(self.m_input_folder)
                # Create output path with same relative structure
                output_path = self.m_output_folder / relative_path.with_suffix('.json')

                # Convert predictions to LabelMe format
                labelme_data = self._convert_to_labelme_format(predictions, image_path, class_names, output_path)
                
                # Save LabelMe JSON
                self._save_labelme_json(labelme_data, output_path)
            
            return True
            
        except Exception as e:
            self.m_logger.error(f"Error processing image ibatch: {e}")
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

            batch_size = max(1, len(image_files) // self.m_jobsCount)
            batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
            # Create and start jobsthreads
            jobsthreads = []
            for i, ibatch in enumerate(batches):
                onethread = threading.Thread(target=self.process_images_inBatch, args=(i, ibatch,))
                jobsthreads.append(onethread)
                onethread.start()
                self.m_logger.info(f"Started onethread {i+1} to process {len(ibatch)} images")
            
            # Wait for all jobsthreads to complete
            for i, ithread in enumerate(jobsthreads):
                ithread.join()
                if AutoAnnotator.exit_event.is_set():
                    self.m_logger.info(f"Thread {i+1} terminated due to exit signal")
                else:
                    self.m_logger.info(f"Thread {i+1} completed")
            
            if AutoAnnotator.exit_event.is_set():
                self.m_logger.info("Processing interrupted by user signal")
                return False
            
            self.m_logger.info(f"Processed {len(image_files)} images")
            return True
            
        except Exception as e:
            self.m_logger.error(f"Error processing images in {self.m_input_folder}: {e}")
            return False


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    logger = get_logger()
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    AutoAnnotator.exit_event.set()


def main():
    """Main function to run the auto annotation tool."""   
    # Initialize logger
    logger = initLogger("auto_annotate")
    logger.info("Auto Annotate Tool started")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Auto annotate images using a CNN model')
    parser.add_argument('--model_file', required=True, help='Path to the model file')
    parser.add_argument('--model_type', required=True, choices=['segmentation', 'detection'], 
                        help='Type of model (segmentation or detection)')
    parser.add_argument('--input_folder', required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_folder', required=True, help='Path to the folder where results should be saved')
    parser.add_argument('--label_name', default=None, help='Label name to use for all annotations (optional, uses model class names if available)')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel jobs for processing images (default: 1)')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    model_file = Path(args.model_file)
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    # Create annotator instance
    annotator = AutoAnnotator(str(model_file), args.model_type, input_folder, output_folder, args.label_name, args.jobs)
    
    # Process images
    logger.info("Starting auto annotation process...")
    success = annotator.process_images()
    
    if success:
        logger.info("Auto annotation process completed successfully.")
        return 0
    else:
        if AutoAnnotator.exit_event.is_set():
            logger.info("Auto annotation process interrupted by user signal.")
        else:
            logger.error("Auto annotation process failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())