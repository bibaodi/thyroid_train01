
import argparse
import pathlib
import csv
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import glog
from typing import Set, Dict
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Verify spreadsheet items exist in dataset')
    parser.add_argument('spreadsheet', type=str, help='Path to spreadsheet file')
    parser.add_argument('dataset', type=str, help='Path to dataset directory')
    return parser.parse_args()

def load_spreadsheet_uids(spreadsheet_path: str) -> Set[str]:
    """Load SOP_UIDs from spreadsheet with case insensitivity"""
    try:
        df = pd.read_excel(spreadsheet_path, sheet_name='verify_3000_tirads1_5')
        return set(df['sop_uid'].astype(str).str.lower().unique())
    except Exception as e:
        glog.get_logger().error(f"Failed to load spreadsheet: {str(e)}")
        raise

def build_dataset_index(dataset_path: str) -> Dict[str, str]:
    """Build index of {uid_lowercase: original_filename} from dataset"""
    index = {}
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            try:
                uid = os.path.splitext(filename)[0].split('_')[0]
                index[uid.lower()] = filename  # Store original case filename
            except Exception as e:
                glog.get_logger().warning(f"Error processing {filename}: {str(e)}")
    return index

def main():
    args = parse_arguments()
    
    if not pathlib.Path(args.spreadsheet).exists():
        glog.get_logger().error(f"Spreadsheet not found: {args.spreadsheet}")
        return
    
    try:
        spreadsheet_uids = load_spreadsheet_uids(args.spreadsheet)
        dataset_index = build_dataset_index(args.dataset)
        
        with open('spreadItemInDataset_validation_report.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SOP_UID', 'Status', 'Matching_Filename'])
            
            for uid in sorted(spreadsheet_uids):
                match = dataset_index.get(uid.lower())
                if match:
                    writer.writerow([uid, 'FOUND', match])
                else:
                    writer.writerow([uid, 'MISSING', ''])
                    glog.get_logger().warning(f"Missing dataset entry for SOP_UID: {uid}")
                    
    except Exception as e:
        glog.get_logger().error(f"Validation failed: {str(e)}")

if __name__ == "__main__":
    glog.glogger = glog.initLogger("findSpreadItemsINFolder.log")
    main()
