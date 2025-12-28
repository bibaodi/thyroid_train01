#!/usr/bin/env python3
"""
SOP4 Training Images Creation Script

This script creates and prepares the SOP4 training dataset by:
1. Cropping images to ultrasound regions using YOLOv11 detector
2. Extracting nodule images using YOLOv11 segmentation model

Processing Features:
- Normal Mode: Saves cropped nodule regions (2W x 2H expanded areas) to nodule_images/
- Debug Mode: Saves full images with green contours and red expanded boxes to nodule_images_debug/
- debug_limit: Limits processing count in both modes (set to -1 for no limit)
- GPU Acceleration: Automatic MPS (Apple Silicon) / CUDA / CPU device detection

Requirements:
- ultralytics (for YOLOv11 models)
- opencv-python
- pandas
- numpy
- torch
- tqdm
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional
import argparse
from ultralytics import YOLO
import warnings

# Suppress ultralytics warnings that might interfere with progress bar
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
os.environ['YOLO_VERBOSE'] = 'False'

# Device detection for optimal performance
def get_device():
    """Detect and return the best available device for model inference"""
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon MPS GPU acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA GPU acceleration")
    else:
        device = "cpu"
        logger.info("Using CPU (no GPU acceleration available)")
    return device

# Add project root to Python path
project_root = "/Users/mouxiaoyong/Documents/PycharmProject/us_feature_classify_v2"
sys.path.append(project_root)

# ======================== CONFIGURATION SECTION ========================
# All configurable parameters are centralized here for easy management

# Dataset and file paths (relative to project root)
CONFIG = {
    # Input data - 独立验证集
    'sop4_filter': 'data/dataset_sop7/all_verify_sop.csv',  # 独立验证集CSV
    'project_root': '/Users/mouxiaoyong/Documents/PycharmProject/us_feature_classify_v2',
    
    # Image directories
    'image_root': '/Users/Shared/tars/verify_images',
    'us_image_root': '/Users/Shared/tars/us_images', 
    'nodule_image_root': '/Users/Shared/tars/nodule_images',
    
    # Log files - 使用独立的日志文件
    'fail_log': 'data/dataset_sop4/verify_fail_log.txt',
    'fail_nodule_log': 'data/dataset_sop4/verify_fail_nodule_log.txt',
    
    # Model paths (YOLOv11 models)
    'yolo_us_nodule_segment_model': 'models/inference/nodule_us_segment_model_v3_best.pt',
    'yolo_us_region_detector': 'models/inference/us_region_detector_v2_best.pt',
    
    # Processing parameters
    'batch_size': 32,
    'nodule_expansion_ratio': 0.5,  # 50% expansion on each side
    'debug_mode': False,
    'debug_limit': -1,  # Set to -1 for no limit
}

# ======================== LOGGING SETUP ========================

def setup_logging():
    """Setup logging configuration"""
    log_level = logging.DEBUG if CONFIG['debug_mode'] else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sop4_training_creation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ======================== UTILITY FUNCTIONS ========================

def create_directories(directories: List[str]) -> None:
    """Create necessary directories if they don't exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the SOP4 filtered dataset"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded dataset with {len(df)} records from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset from {csv_path}: {str(e)}")
        raise

def check_model_files() -> bool:
    """Check if required model files exist"""
    models_to_check = [
        CONFIG['yolo_us_region_detector'],
        CONFIG['yolo_us_nodule_segment_model']
    ]
    
    missing_models = []
    for model_path in models_to_check:
        full_path = os.path.join(CONFIG['project_root'], model_path)
        if not os.path.exists(full_path):
            missing_models.append(model_path)
    
    if missing_models:
        logger.error(f"Missing model files: {missing_models}")
        return False
    
    logger.info("All required model files found")
    return True

# ======================== YOLO MODEL LOADING ========================

def load_yolo_models():
    """Load YOLOv11 models for region detection and nodule segmentation using ultralytics"""
    try:
        # Get the best available device for inference
        device = get_device()
        
        # Load US region detector (YOLOv11) with reduced verbosity and device specification
        us_detector_path = os.path.join(CONFIG['project_root'], CONFIG['yolo_us_region_detector'])
        us_detector = YOLO(us_detector_path, verbose=False)
        us_detector.to(device)  # Move model to the specified device
        logger.info(f"Loaded YOLOv11 US region detector from {us_detector_path} on {device}")
        
        # Load nodule segmentation model (YOLOv11) with reduced verbosity and device specification
        nodule_segmenter_path = os.path.join(CONFIG['project_root'], CONFIG['yolo_us_nodule_segment_model'])
        nodule_segmenter = YOLO(nodule_segmenter_path, verbose=False)
        nodule_segmenter.to(device)  # Move model to the specified device
        logger.info(f"Loaded YOLOv11 nodule segmenter from {nodule_segmenter_path} on {device}")
        
        return us_detector, nodule_segmenter, device
        
    except Exception as e:
        logger.error(f"Failed to load YOLOv11 models: {str(e)}")
        raise

# ======================== IMAGE PROCESSING FUNCTIONS ========================

def process_us_region_detection(df: pd.DataFrame, us_detector, device: str, fail_log_path: str) -> Tuple[int, int]:
    """
    Process images to extract ultrasound regions
    
    Args:
        df: DataFrame with sop_uid and access_no columns
        us_detector: YOLO model for US region detection
        device: Device for inference (mps/cuda/cpu)
        fail_log_path: Path to failure log file
        
    Returns:
        Tuple of (successful_count, total_count)
    """
    logger.info("Starting US region detection processing...")
    
    # Limit processing if debug_limit is set (works in both normal and debug mode)
    if CONFIG['debug_limit'] > 0:
        df = df.head(CONFIG['debug_limit'])
        mode_text = "DEBUG MODE" if CONFIG['debug_mode'] else "LIMITED MODE"
        logger.info(f"{mode_text}: Processing only first {CONFIG['debug_limit']} images")
    else:
        logger.info("Processing all images in dataset")
    
    successful_count = 0
    failed_images = []
    
    # Process images in batches
    for i in tqdm(range(0, len(df), CONFIG['batch_size']), 
                  desc="Processing US regions", 
                  unit="batch", 
                  leave=True, 
                  ncols=80):
        batch_df = df.iloc[i:i+CONFIG['batch_size']]
        batch_images = []
        batch_info = []
        
        # Load batch of images
        for _, row in batch_df.iterrows():
            sop_uid = row['sop_uid']
            access_no = row['access_no']
            
            input_path = os.path.join(CONFIG['image_root'], str(access_no), f"{sop_uid}.jpg")
            output_dir = os.path.join(CONFIG['us_image_root'], str(access_no))
            output_path = os.path.join(output_dir, f"{sop_uid}.jpg")
            
            # Skip if output already exists
            if os.path.exists(output_path):
                successful_count += 1
                continue
                
            # Check if input image exists
            if not os.path.exists(input_path):
                failed_images.append(f"{sop_uid},{access_no},Input image not found: {input_path}")
                continue
                
            try:
                image = cv2.imread(input_path)
                if image is None:
                    failed_images.append(f"{sop_uid},{access_no},Failed to read image: {input_path}")
                    continue
                    
                batch_images.append(image)
                batch_info.append({
                    'sop_uid': sop_uid,
                    'access_no': access_no,
                    'output_dir': output_dir,
                    'output_path': output_path
                })
                
            except Exception as e:
                failed_images.append(f"{sop_uid},{access_no},Error loading image: {str(e)}")
        
        # Process batch with YOLO (suppress output to avoid interfering with progress bar)
        if batch_images:
            try:
                results = us_detector(batch_images, verbose=False, device=device)
                
                for idx, result in enumerate(results):
                    info = batch_info[idx]
                    
                    try:
                        # Get the best detection (highest confidence)
                        if len(result.boxes) > 0:
                            # Sort by confidence and take the best one
                            confidences = result.boxes.conf.cpu().numpy()
                            best_idx = np.argmax(confidences)
                            
                            # Get bounding box coordinates
                            box = result.boxes.xyxy[best_idx].cpu().numpy()
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Crop the ultrasound region
                            us_region = batch_images[idx][y1:y2, x1:x2]
                            
                            # Create output directory and save
                            Path(info['output_dir']).mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(info['output_path'], us_region)
                            successful_count += 1
                            
                        else:
                            failed_images.append(f"{info['sop_uid']},{info['access_no']},No US region detected")
                            
                    except Exception as e:
                        failed_images.append(f"{info['sop_uid']},{info['access_no']},Error processing detection: {str(e)}")
                        
            except Exception as e:
                for info in batch_info:
                    failed_images.append(f"{info['sop_uid']},{info['access_no']},Batch processing error: {str(e)}")
    
    # Write failure log
    if failed_images:
        with open(fail_log_path, 'w') as f:
            f.write("sop_uid,access_no,error_message\n")
            for failed_img in failed_images:
                f.write(failed_img + "\n")
        logger.warning(f"Logged {len(failed_images)} failed images to {fail_log_path}")
    
    total_count = len(df)
    success_rate = (successful_count / total_count) * 100 if total_count > 0 else 0
    
    logger.info(f"US region detection completed: {successful_count}/{total_count} images processed successfully ({success_rate:.2f}%)")
    
    return successful_count, total_count

def process_nodule_extraction(df: pd.DataFrame, nodule_segmenter, device: str, fail_log_path: str) -> Tuple[int, int]:
    """
    Process images to extract nodule regions with expansion
    In debug mode, visualizes contours and expanded boxes on the original image
    
    Args:
        df: DataFrame with sop_uid and access_no columns
        nodule_segmenter: YOLO model for nodule segmentation
        device: Device for inference (mps/cuda/cpu)
        fail_log_path: Path to failure log file
        
    Returns:
        Tuple of (successful_count, total_count)
    """
    logger.info("Starting nodule extraction processing...")
    
    # Limit processing if debug_limit is set (works in both normal and debug mode)
    if CONFIG['debug_limit'] > 0:
        df = df.head(CONFIG['debug_limit'])
        mode_text = "DEBUG MODE" if CONFIG['debug_mode'] else "LIMITED MODE"
        vis_text = " with visualization" if CONFIG['debug_mode'] else ""
        logger.info(f"{mode_text}: Processing only first {CONFIG['debug_limit']} images{vis_text}")
    else:
        logger.info("Processing all images in dataset")
    
    successful_count = 0
    failed_images = []
    
    # Process images in batches
    for i in tqdm(range(0, len(df), CONFIG['batch_size']), 
                  desc="Processing nodule extraction", 
                  unit="batch", 
                  leave=True, 
                  ncols=80):
        batch_df = df.iloc[i:i+CONFIG['batch_size']]
        batch_images = []
        batch_info = []
        
        # Load batch of images
        for _, row in batch_df.iterrows():
            sop_uid = row['sop_uid']
            access_no = row['access_no']
            
            input_path = os.path.join(CONFIG['us_image_root'], str(access_no), f"{sop_uid}.jpg")
            
            # Use different output directory for debug mode visualization
            if CONFIG['debug_mode']:
                output_dir = os.path.join(CONFIG['nodule_image_root'] + "_debug", str(access_no))
                output_path = os.path.join(output_dir, f"{sop_uid}.jpg")
            else:
                output_dir = os.path.join(CONFIG['nodule_image_root'], str(access_no))
                output_path = os.path.join(output_dir, f"{sop_uid}.jpg")
            
            # Skip if output already exists
            if os.path.exists(output_path):
                successful_count += 1
                continue
                
            # Check if input image exists
            if not os.path.exists(input_path):
                failed_images.append(f"{sop_uid},{access_no},US region image not found: {input_path}")
                continue
                
            try:
                image = cv2.imread(input_path)
                if image is None:
                    failed_images.append(f"{sop_uid},{access_no},Failed to read US image: {input_path}")
                    continue
                    
                batch_images.append(image)
                batch_info.append({
                    'sop_uid': sop_uid,
                    'access_no': access_no,
                    'output_dir': output_dir,
                    'output_path': output_path,
                    'image_shape': image.shape
                })
                
            except Exception as e:
                failed_images.append(f"{sop_uid},{access_no},Error loading US image: {str(e)}")
        
        # Process batch with YOLO (suppress output to avoid interfering with progress bar)
        if batch_images:
            try:
                results = nodule_segmenter(batch_images, verbose=False, device=device)
                
                for idx, result in enumerate(results):
                    info = batch_info[idx]
                    
                    try:
                        # Get the best segmentation (highest confidence)
                        if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                            # Sort by confidence and take the best one
                            confidences = result.boxes.conf.cpu().numpy()
                            best_idx = np.argmax(confidences)
                            
                            # Get mask for the best detection
                            mask = result.masks.data[best_idx].cpu().numpy()
                            
                            # Get original image dimensions
                            original_height, original_width = info['image_shape'][:2]
                            
                            # Handle YOLO letterbox coordinate transformation
                            # YOLO typically resizes to 640x640 with letterbox padding
                            model_input_size = 640  # YOLO input size
                            
                            # Calculate the scale factor (keep aspect ratio)
                            scale = min(model_input_size / original_width, model_input_size / original_height)
                            
                            # Calculate the actual resized dimensions (before padding)
                            new_width = int(original_width * scale)
                            new_height = int(original_height * scale)
                            
                            # Calculate padding offsets
                            pad_x = (model_input_size - new_width) // 2
                            pad_y = (model_input_size - new_height) // 2
                            
                            if CONFIG['debug_mode']:
                                logger.debug(f"Original: {original_width}x{original_height}, Mask: {mask.shape}")
                                logger.debug(f"Scale: {scale:.3f}, New size: {new_width}x{new_height}")
                                logger.debug(f"Padding: x={pad_x}, y={pad_y}")
                            
                            # Extract the valid region from the mask (remove padding)
                            mask_valid = mask[pad_y:pad_y+new_height, pad_x:pad_x+new_width]
                            
                            # Resize the valid mask region to match original image dimensions
                            mask_resized = cv2.resize(mask_valid, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                            
                            # Find contours from the properly sized mask
                            mask_uint8 = (mask_resized * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                # Get the largest contour (most likely the nodule)
                                largest_contour = max(contours, key=cv2.contourArea)
                                
                                # Get bounding rectangle
                                x, y, w, h = cv2.boundingRect(largest_contour)
                                
                                if CONFIG['debug_mode']:
                                    logger.debug(f"Nodule bounding box: x={x}, y={y}, w={w}, h={h}")
                                    logger.debug(f"Contour area: {cv2.contourArea(largest_contour)}")
                                    logger.debug(f"Contour center: {(x + w//2, y + h//2)}")
                                
                                # Expand the bounding box by 50% on each side
                                expansion = CONFIG['nodule_expansion_ratio']
                                expand_w = int(w * expansion)
                                expand_h = int(h * expansion)
                                
                                # Calculate expanded coordinates
                                x1 = max(0, x - expand_w)
                                y1 = max(0, y - expand_h)
                                x2 = min(info['image_shape'][1], x + w + expand_w)
                                y2 = min(info['image_shape'][0], y + h + expand_h)
                                
                                if CONFIG['debug_mode']:
                                    logger.debug(f"Expanded coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                    logger.debug(f"Expanded size: {x2-x1}x{y2-y1} (original: {w}x{h})")
                                
                                # Create output directory
                                Path(info['output_dir']).mkdir(parents=True, exist_ok=True)
                                
                                if CONFIG['debug_mode']:
                                    # DEBUG MODE: Draw contours and expanded box on original image
                                    debug_image = batch_images[idx].copy()
                                    
                                    # Draw green contour
                                    cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 2)
                                    
                                    # Draw red expanded bounding box
                                    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    
                                    # Add text labels
                                    cv2.putText(debug_image, 'Nodule Contour', (x1, y1-20), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    cv2.putText(debug_image, f'Expanded Box (2W x 2H)', (x1, y2+20), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    
                                    # Save the visualized image
                                    cv2.imwrite(info['output_path'], debug_image)
                                    
                                else:
                                    # NORMAL MODE: Crop and save the expanded nodule region
                                    nodule_region = batch_images[idx][y1:y2, x1:x2]
                                    cv2.imwrite(info['output_path'], nodule_region)
                                
                                successful_count += 1
                                
                            else:
                                failed_images.append(f"{info['sop_uid']},{info['access_no']},No contours found in segmentation mask")
                                
                        else:
                            failed_images.append(f"{info['sop_uid']},{info['access_no']},No nodule segmentation detected")
                            
                    except Exception as e:
                        failed_images.append(f"{info['sop_uid']},{info['access_no']},Error processing segmentation: {str(e)}")
                        
            except Exception as e:
                for info in batch_info:
                    failed_images.append(f"{info['sop_uid']},{info['access_no']},Batch processing error: {str(e)}")
    
    # Write failure log
    if failed_images:
        with open(fail_log_path, 'w') as f:
            f.write("sop_uid,access_no,error_message\n")
            for failed_img in failed_images:
                f.write(failed_img + "\n")
        logger.warning(f"Logged {len(failed_images)} failed images to {fail_log_path}")
    
    total_count = len(df)
    success_rate = (successful_count / total_count) * 100 if total_count > 0 else 0
    
    if CONFIG['debug_mode']:
        logger.info(f"Nodule extraction completed (DEBUG MODE - with visualization): {successful_count}/{total_count} images processed successfully ({success_rate:.2f}%)")
        logger.info("DEBUG MODE: Saved images contain green contours and red expanded boxes for visualization")
        logger.info(f"DEBUG MODE: Images saved to {CONFIG['nodule_image_root']}_debug/")
    else:
        logger.info(f"Nodule extraction completed: {successful_count}/{total_count} images processed successfully ({success_rate:.2f}%)")
    
    return successful_count, total_count

# ======================== MAIN EXECUTION ========================

def main():
    """Main execution function"""
    global logger
    logger = setup_logging()
    
    logger.info("Starting SOP4 training images creation process")
    logger.info(f"Debug mode: {'ON' if CONFIG['debug_mode'] else 'OFF'}")
    
    try:
        # Check if required model files exist
        if not check_model_files():
            raise Exception("Required model files are missing")
        
        # Create necessary directories
        directories_to_create = [
            CONFIG['us_image_root'],
            CONFIG['nodule_image_root'],
            os.path.dirname(os.path.join(CONFIG['project_root'], CONFIG['fail_log'])),
            os.path.dirname(os.path.join(CONFIG['project_root'], CONFIG['fail_nodule_log']))
        ]
        
        # Add debug directory if in debug mode
        if CONFIG['debug_mode']:
            directories_to_create.append(CONFIG['nodule_image_root'] + "_debug")
            logger.info("Debug mode: Creating separate debug output directory")
        
        create_directories(directories_to_create)
        logger.info("Created necessary directories")
        
        # Load dataset
        sop4_filter_path = os.path.join(CONFIG['project_root'], CONFIG['sop4_filter'])
        df = load_dataset(sop4_filter_path)
        
        # Load YOLO models with device acceleration
        us_detector, nodule_segmenter, device = load_yolo_models()
        
        # Process US region detection
        logger.info("=" * 60)
        logger.info("STEP 1: US Region Detection and Cropping")
        logger.info("=" * 60)
        
        fail_log_path = os.path.join(CONFIG['project_root'], CONFIG['fail_log'])
        us_success_count, us_total_count = process_us_region_detection(df, us_detector, device, fail_log_path)
        
        # Process nodule extraction
        logger.info("=" * 60)
        logger.info("STEP 2: Nodule Extraction and Expansion")
        logger.info("=" * 60)
        
        fail_nodule_log_path = os.path.join(CONFIG['project_root'], CONFIG['fail_nodule_log'])
        nodule_success_count, nodule_total_count = process_nodule_extraction(df, nodule_segmenter, device, fail_nodule_log_path)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"US Region Detection: {us_success_count}/{us_total_count} ({(us_success_count/us_total_count)*100:.2f}%) successful")
        logger.info(f"Nodule Extraction: {nodule_success_count}/{nodule_total_count} ({(nodule_success_count/nodule_total_count)*100:.2f}%) successful")
        logger.info(f"Debug mode: {'ON' if CONFIG['debug_mode'] else 'OFF'}")
        
        # Processing limit information
        if CONFIG['debug_limit'] > 0:
            logger.info(f"Note: Limited processing to {CONFIG['debug_limit']} images per step")
        else:
            logger.info("Note: Processed all images in dataset")
            
        # Mode-specific information
        if CONFIG['debug_mode']:
            logger.info("Note: Debug mode saved images with green contours and red expanded boxes for visualization")
            logger.info(f"Note: Debug visualization images saved to: {CONFIG['nodule_image_root']}_debug/")
        else:
            logger.info("Note: Normal mode saved cropped nodule regions")
            logger.info(f"Note: Cropped images saved to: {CONFIG['nodule_image_root']}/")

        
        logger.info("SOP4 training images creation completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
