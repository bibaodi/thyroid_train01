#screenTiRADS45Bethesda26FromDataset.py

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from SpreadSheetChecker import SpreadSheetChecker 
from findDuplicateFileByUIDBasedFilename import DuplicateFileFinder 
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
    # Initialize sschecker
    # sschecker = SpreadSheetChecker(
    #     file_path="2223multi_nodule_45434.xlsx",
    #     sheet_name="origintable",
    #     columns=["sop_uid", "ti_rads", "bethesda"],
    #     primary_key="sop_uid"
    # )
    return sschecker

def main_screenTirads45Bethesda26():
    args = parse_arguments()

    # Validate spreadsheet exists early
    if not pathlib.Path(args.spreadsheet).exists():
        glog.get_logger().error(f"Spreadsheet file not found: {args.spreadsheet}")
        return

    finder = DuplicateFileFinder(
        args.source_dir,
        args.reference_dir,
        args.output,
        args.match_method
    )
    nameIndex =  finder._create_file_index(args.source_dir)

    sschecker = getSpreadSheetChecker(args.spreadsheet)
    if sschecker.error:
        glog.get_logger().error("SpreadSheetChecker initialization failed")
        return
    
    # New CSV output
    with open('screened_tirads_bethesda_records.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['UID', 'tirads', 'bethesda'])
        
        for iUID, irealpath in nameIndex.items():
            try:
                suid = iUID.split("_")[0] if "_" in iUID else iUID
                row = sschecker.find_target_row(suid)
                
                if not row:
                    glog.get_logger().debug(f"Target not found in spreadsheet: {iUID}")
                    continue

                # Validate and convert types
                ti_rads = int(sschecker.get_item_value(row, "ti_rads"))
                bethesda = int(sschecker.get_item_value(row, "bethesda"))
                
                if ti_rads not in {4, 5} or bethesda not in {2, 6}:
                    continue
                
                # Write validated integers
                writer.writerow([iUID, ti_rads, bethesda])
                glog.get_logger().info(f"Valid case: {iUID} TiRADS={ti_rads}, Bethesda={bethesda}")

            except (ValueError, TypeError) as e:
                glog.get_logger().warning(f"Invalid data for {iUID}: {str(e)}")
            except Exception as e:
                glog.get_logger().error(f"Error processing {iUID}: {str(e)}")
        else:
            glog.get_logger().warning(f"Target not found:[{iUID}]")
    
    
if __name__ == '__main__':
    try:
        glog.glogger = glog.initLogger("screenTiRADS45Bethesda26FromDataset.log")
        main_screenTirads45Bethesda26()
    except KeyboardInterrupt:
        glog.get_logger().info("Operation cancelled by user")
    except Exception as error:
        glog.get_logger().error(f"Critical failure: {error}")
        raise SystemExit(1) from error
