#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert image files with versioned JSON annotations to Labelme format.

This script processes image files where each image has a corresponding folder containing 
multiple versions of labeled JSON files. It selects the latest versioned JSON file 
for conversion to Labelme format.

Author: eton
Date: 2025-08-19
Version: 1.0
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to sys.path to import LabelmeJson
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import LabelmeJson


def get_latest_xbjson_file(json_folder):
    """
    Get the latest versioned JSON file from a folder based on timestamp in filename.
    
    Args:
        json_folder (Path): Path to the folder containing JSON files
        
    Returns:
        Path: Path to the latest JSON file, or None if no JSON files found
    """
    json_files = list(json_folder.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {json_folder}")
        return None
    
    # Extract timestamps from filenames and find the latest
    latest_file = None
    latest_timestamp = None
    
    for json_file in json_files:
        try:
            # Extract timestamp from filename (assuming format: [ID].[TIMESTAMP]MARK.json)
            filename = json_file.stem  # Remove .json extension
            parts = filename.split(".")
            if len(parts) >= 2 and "MARK" in parts[-1]:
                timestamp_str = parts[-1].replace("MARK", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_file = json_file
        except ValueError as e:
            logger.warning(f"Could not parse timestamp from {json_file.name}: {e}")
            continue
    
    return latest_file


def convert_to_labelme_format(json_data, image_path, label_name="lesion"):
    """
    Convert JSON annotation data to Labelme format.
    
    Args:
        json_data (dict): Original JSON annotation data
        image_path (Path): Path to the image file
        label_name (str): Label name to use in the Labelme JSON
        
    Returns:
        dict: Labelme format data
    """
    # Create a new Labelme target object
    labelme_data = LabelmeJson.getOneTargetObj()
    labelme_data["imagePath"] = image_path.name
    labelme_data["imageData"] = None
    
    # Clear existing shapes
    labelme_data["shapes"].clear()
    
    # Process lesions in the JSON data
    if "data" in json_data and len(json_data["data"]) > 0:
        lesions_data = json_data["data"][0].get("lesions", [])
        
        for i, lesion in enumerate(lesions_data):
            # Create a new shape object
            shape_obj = LabelmeJson.getOneShapeObj()
            shape_obj["label"] = label_name  # Use the specified label name
            
            # Extract polygon points
            if "polygonPoint" in lesion:
                points = lesion["polygonPoint"]
                # Filter out invalid points
                valid_points = [point for point in points if point[0] is not None and point[0] >= 10]
                
                if len(valid_points) >= 4:  # Labelme polygons need at least 4 points
                    shape_obj["points"] = valid_points
                    labelme_data["shapes"].append(shape_obj)
                else:
                    logger.warning(f"Not enough valid points for lesion {i} in {image_path.name}")
            else:
                logger.warning(f"No polygon points found for lesion {i} in {image_path.name}")
    
    return labelme_data


def process_image_folder(image_path, output_dir, label_name="lesion"):
    """
    Process an image folder to convert annotations to Labelme format.
    
    Args:
        image_path (Path): Path to the image file
        output_dir (Path): Directory to save the converted Labelme JSON
        label_name (str): Label name to use in the Labelme JSON
    """
    # Find folders that start with the image's stem
    json_folder_path = None
    for item in image_path.parent.iterdir():
        if item.is_dir() and item.name.startswith(image_path.stem):
            json_folder_path = item
            break
    
    if json_folder_path is None:
        logger.warning(f"No JSON folder found for {image_path}")
        return
    
    # Get the latest JSON file
    latest_json = get_latest_xbjson_file(json_folder_path)
    
    if not latest_json:
        logger.warning(f"No valid JSON file found in {json_folder_path}")
        return
    
    # Read the JSON data
    try:
        with open(latest_json, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {latest_json}: {e}")
        return
    
    # Convert to Labelme format
    labelme_data = convert_to_labelme_format(json_data, image_path, label_name)
    
    # Save the converted data
    output_file = output_dir / f"{image_path.stem}.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Converted {image_path.name} to Labelme format: {output_file}")
        
        # Also copy the image file to the output directory
        output_image = output_dir / image_path.name
        with open(image_path, 'rb') as src, open(output_image, 'wb') as dst:
            dst.write(src.read())
        logger.info(f"Copied image {image_path.name} to {output_image}")
    except Exception as e:
        logger.error(f"Error saving Labelme file {output_file}: {e}")


def main(input_dir, output_dir, label_name="lesion"):
    """
    Main function to process all image files in the input directory.
    
    Args:
        input_dir (str): Path to the input directory containing images and JSON folders
        output_dir (str): Path to the output directory for Labelme JSON files
        label_name (str): Label name to use in the Labelme JSON
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all image files in the input directory using LabelmeJson function
    image_files = LabelmeJson.getImageFilesBySuffixes(input_path)
    
    # Process all subdirectories as well
    for image_file in image_files:
        # Skip if already in a labelme output directory
        if 'labelme' in str(image_file):
            continue
        logger.info(f"Processing {image_file.name}")
        process_image_folder(image_file, output_path, label_name)
    
    logger.info("Conversion completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image files with versioned JSON annotations to Labelme format.")
    parser.add_argument("input_dir", help="Path to the input directory containing images and JSON folders")
    parser.add_argument("output_dir", help="Path to the output directory for Labelme JSON files")
    parser.add_argument("--label-name", default="lesion", help="Label name to use in the Labelme JSON (default: lesion)")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir, args.label_name)
