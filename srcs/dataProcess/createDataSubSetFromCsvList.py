#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##eton@20250701 this script is used to create a subset of data from a list of CSV files which contains filenames and their corresponding labels.
"""
Image Organizer v1.0
Author: Eton
Created: 250701
Description: Organizes medical images based on CSV file which contains filenames and their corresponding labels.
"""

import argparse
import pathlib
import pandas as pd
import shutil
import sys, os
from typing import Dict, Tuple, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import glog

class ImageOrganizer:
    def __init__(self, source_root: pathlib.Path, output_root: pathlib.Path):
        self.source_root = source_root.resolve()
        self.output_root = output_root.resolve()
        self.file_index = self._build_file_index()
        glog.get_logger().info(f"Indexed {len(self.file_index)} source files")

    def _build_file_index(self) -> Dict[str, pathlib.Path]:
        """Create filename to path mapping for all files in source root"""
        index = {}
        for file_path in self.source_root.rglob('*'):
            if file_path.is_file():
                index[file_path.name] = file_path
        return index

    def process_csv(self, csv_path: pathlib.Path) -> List[Tuple[str, str]]:
        """Read and validate CSV file"""
        try:
            df = pd.read_csv(csv_path, header=None, names=['filename', 'label'])
            return list(df[['filename', 'label']].itertuples(index=False, name=None))
        except Exception as e:
            glog.get_logger().error(f"CSV read error: {e}")
            raise

    def copy_files(self, file_list: List[Tuple[str, str]]):
        """Main copy operation with logging"""
        stats = {'copied': 0, 'skipped': 0, 'errors': 0}
        
        for filename, label in file_list:
            source_path = self.file_index.get(filename)
            if not self._validate_file(filename, source_path, stats):
                continue
                
            target_dir = self.output_root / label
            if not self._create_target_dir(target_dir, stats):
                continue
                
            self._perform_copy(source_path, target_dir / filename, filename, stats)
            
        glog.get_logger().info(f"Operation complete: {stats['copied']} copied, "
                             f"{stats['skipped']} skipped, {stats['errors']} errors")

    def _validate_file(self, filename, source_path, stats) -> bool:
        if not source_path:
            glog.get_logger().warning(f"File not found: {filename}")
            stats['skipped'] += 1
            return False
        return True

    def _create_target_dir(self, target_dir, stats) -> bool:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            glog.get_logger().error(f"Directory creation failed: {e}")
            stats['errors'] += 1
            return False

    def _perform_copy(self, source, target, filename, stats):
        try:
            shutil.copy2(source, target)
            glog.get_logger().info(f"Copied {filename} to {target}")
            stats['copied'] += 1
        except Exception as e:
            glog.get_logger().error(f"Copy failed for {filename}: {e}")
            stats['errors'] += 1

def main():
    parser = argparse.ArgumentParser(description='Organize medical images by labels')
    parser.add_argument('source_root', type=pathlib.Path, help='Source images directory')
    parser.add_argument('csv_file', type=pathlib.Path, help='CSV file with filenames and labels')
    parser.add_argument('output_root', type=pathlib.Path, help='Output directory')
    parser.add_argument('--log_file', type=pathlib.Path, default='image_organizer.log',
                       help='Path to log file')
    
    args = parser.parse_args()

    # Validate input parameters
    if not args.source_root.exists() or not args.source_root.is_dir():
        glog.get_logger().error(f"Invalid source directory: {args.source_root}")
        sys.exit(1)
    if not args.csv_file.exists() or not args.csv_file.is_file():
        glog.get_logger().error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    args.output_root.mkdir(parents=True, exist_ok=True)

    try:
        organizer = ImageOrganizer(args.source_root, args.output_root)
        file_list = organizer.process_csv(args.csv_file)
        organizer.copy_files(file_list)
    except Exception as e:
        glog.get_logger().error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    glog.glogger = glog.initLogger("image_organizer.log")
    main()
