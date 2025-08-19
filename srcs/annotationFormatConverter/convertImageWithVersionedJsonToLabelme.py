#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert image files with versioned JSON annotations to Labelme format.
This script traverses a directory of images, finds corresponding JSON files in 
versioned subdirectories, selects the latest version, and converts annotations 
to Labelme JSON format.

Author: eton
Date: 2025-08-13
Version: 1.0
"""

import os
import json
import pathlib
import shutil
import re
from datetime import datetime
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LabelmeJson module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import LabelmeJson


def select_latest_json(json_dir):
    """Select the latest versioned JSON file from a directory."""
    try:
        json_files = list(json_dir.glob('*.json'))
        if not json_files:
            logger.warning(f"No JSON files found in {json_dir}")
            return None
        
        # Sort by modification time, most recent first
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return json_files[0]
    except Exception as e:
        logger.error(f"Error selecting latest JSON in {json_dir}: {e}")
        return None


def convert_to_labelme_format(json_data, image_filename):
    """Convert custom JSON format to Labelme format."""
    try:
        # Create a new Labelme target object
        labelme_obj = LabelmeJson.getOneTargetObj()
        labelme_obj["imagePath"] = image_filename
        
        # Clear existing shapes
        labelme_obj["shapes"].clear()
        
        # Process lesions in the JSON data
        if "data" in json_data and len(json_data["data"]) > 0:
            lesions_data = json_data["data"][0].get("lesions", [])
            
            for i, lesion in enumerate(lesions_data):
                # Create a new shape object
                shape_obj = LabelmeJson.getOneShapeObj()
                shape_obj["label"] = "lesion"  # Default label
                
                # Extract polygon points
                if "polygonPoint" in lesion:
                    points = lesion["polygonPoint"]
                    # Filter out invalid points
                    valid_points = [point for point in points if point[0] is not None and point[0] >= 10]
                    
                    if len(valid_points) >= 4:  # Labelme polygons need at least 4 points
                        shape_obj["points"] = valid_points
                        labelme_obj["shapes"].append(shape_obj)
                    else:
                        logger.warning(f"Not enough valid points for lesion {i} in {image_filename}")
                else:
                    logger.warning(f"No polygon points found for lesion {i} in {image_filename}")
        
        return labelme_obj
    except Exception as e:
        logger.error(f"Error converting JSON to Labelme format: {e}")
        return None


def process_image_with_json(image_path, json_dir_path):
    """Process an image file with its corresponding JSON annotations."""
    try:
        # Select the latest JSON file
        latest_json = select_latest_json(json_dir_path)
        if not latest_json:
            logger.warning(f"No JSON file found for image {image_path}")
            return False
        
        # Read the JSON file
        with open(latest_json, 'r') as f:
            json_data = json.load(f)
        
        # Convert to Labelme format
        labelme_data = convert_to_labelme_format(json_data, image_path.name)
        if not labelme_data:
            logger.error(f"Failed to convert {latest_json} to Labelme format")
            return False
        
        # Create output directory
        output_dir = image_path.parent / "labelme"
        output_dir.mkdir(exist_ok=True)
        
        # Save the Labelme JSON file
        output_json_path = output_dir / f"{image_path.stem}.json"
        with open(output_json_path, 'w') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        
        # Copy the image file
        output_image_path = output_dir / image_path.name
        shutil.copy2(image_path, output_image_path)
        
        logger.info(f"Successfully converted {image_path.name} to Labelme format")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False


def process_directory(root_dir):
    """Process all images in a directory and its subdirectories."""
    root_path = pathlib.Path(root_dir)
    
    # Common image suffixes
    image_suffixes = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Find all image files
    for suffix in image_suffixes:
        for image_path in root_path.rglob(f'*{suffix}'):
            # Skip images already in labelme directories
            if 'labelme' in str(image_path):
                continue
            
            # Look for corresponding JSON directory
            json_dir_name = f"{image_path.stem}_noname_o"
            json_dir_path = image_path.parent / json_dir_name
            
            if json_dir_path.is_dir():
                logger.info(f"Processing {image_path}")
                process_image_with_json(image_path, json_dir_path)
            else:
                logger.warning(f"No JSON directory found for {image_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert images with versioned JSON annotations to Labelme format.')
    parser.add_argument('input_dir', help='Input directory containing images and JSON annotations')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    logger.info(f"Starting conversion process for directory: {args.input_dir}")
    process_directory(args.input_dir)
    logger.info("Conversion process completed.")

if __name__ == "__main__":
    main()
