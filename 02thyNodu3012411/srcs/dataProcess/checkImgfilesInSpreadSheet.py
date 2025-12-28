#!/bin/env python
#eton@250329 not finished yet.

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from SpreadSheetChecker import SpreadSheetChecker 
import glog
#from  glog import glogger as glog.glogger
glog.glogger = None#glog.glogger
import argparse
import pathlib
import typing
import csv


def parse_arguments() -> argparse.Namespace:
    """Configure and parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Find duplicate images between two directories.')
    parser.add_argument('source_dir', help='Primary directory containing images to check')
    parser.add_argument('-t', '--spreadsheet', required=True, help='Path to medical data spreadsheet')
    parser.add_argument('reference_dir', help='Comparison directory containing reference images')
    parser.add_argument('-o', '--output', default='duplicate-file-matches.csv', help='Output file path for results')
    parser.add_argument('--match-method', choices=['content', 'filename'], default='filename',
                      help='Comparison method: file contents (MD5) or base filename')
    return parser.parse_args()

def getSpreadSheetChecker(
    spreadsheet_path: typing.Union[str, pathlib.Path], 
    sheet_name: str = None,
    columns: list = None,
    primary_key: str = "sop_uid"
) -> SpreadSheetChecker:
    """Initialize and return a SpreadSheetChecker instance with safe defaults"""
    # Convert to None for mutable defaults
    sheet_name = sheet_name or "origintable"
    columns = columns or ["sop_uid", "ti_rads", "bethesda"]
    
    spreadsheet_path = pathlib.Path(spreadsheet_path)
    if not spreadsheet_path.exists():
        raise FileNotFoundError(f"Spreadsheet not found: {spreadsheet_path}")
    if not isinstance(spreadsheet_path, pathlib.Path):
        spreadsheet_path = pathlib.Path(spreadsheet_path)
        if not spreadsheet_path.is_file():
            raise ValueError(f"Invalid spreadsheet path: {spreadsheet_path}")
    
    sschecker = SpreadSheetChecker(spreadsheet_path, sheet_name, columns, primary_key)

    return sschecker


def _create_file_index(directory_path: str) -> Dict[Union[str, bytes], List[str]]:
    """Build file index for a directory"""
    file_index: Dict[Union[str, bytes], List[str]] = {}
    abs_dir_path = os.path.abspath(directory_path)
    parent_dir = os.path.dirname(abs_dir_path)
    
    glog.get_logger().info(f"Building index for: {directory_path} ")

    for root, _, filenames in os.walk(abs_dir_path):
        for filename in filenames:
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension not in IMAGE_EXTENSIONS:
                continue

            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, parent_dir)
            
            try:
                index_key = os.path.splitext(filename)[0].lower() #self._get_file_key(file_path, filename)
                file_index.setdefault(index_key, []).append(relative_path)
            except Exception as error:
                glog.get_logger().warning(f"Skipping unreadable file {file_path}: {error}")
    return file_index

def main_checkIsImgesInSpreadsheet():
    args = parse_arguments()

    # Validate spreadsheet exists early
    if not pathlib.Path(args.spreadsheet).exists():
        glog.get_logger().error(f"Spreadsheet file not found: {args.spreadsheet}")
        return

    nameIndex =  _create_file_index(args.source_dir)

    sschecker = getSpreadSheetChecker(args.spreadsheet, 'verify_3000_tirads1_5')
    if sschecker.error:
        glog.get_logger().error("SpreadSheetChecker initialization failed")
        return
    
    # New CSV output
    with open('imagesInSpread_records.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['UID', 'InSpreadSheet'])
        
        for iUID, irealpath in nameIndex.items():
            try:
                suid = iUID.split("_")[0] if "_" in iUID else iUID
                row = sschecker.find_target_row(suid)
                
                if not row:
                    glog.get_logger().debug(f"Target not found in spreadsheet: {iUID}")
                    continue

                # Write validated integers
                writer.writerow([iUID, args.spreadsheet])
                glog.get_logger().info(f"Valid case: {iUID} found in spread sheet")

            except (ValueError, TypeError) as e:
                glog.get_logger().warning(f"Invalid data for {iUID}: {str(e)}")
            except Exception as e:
                glog.get_logger().error(f"Error processing {iUID}: {str(e)}")
        else:
            glog.get_logger().warning(f"Target not found:[{iUID}]")
    
    
if __name__ == '__main__':
    try:
        glog.glogger = glog.initLogger("check Img in spread sheet.log")
        main_checkIsImgesInSpreadsheet()
    except KeyboardInterrupt:
        glog.get_logger().info("Operation cancelled by user")
    except Exception as error:
        glog.get_logger().error(f"Critical failure: {error}")
        raise SystemExit(1) from error
