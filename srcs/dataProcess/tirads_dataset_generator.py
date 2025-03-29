import os
import argparse
import pathlib
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import glog
from typing import Dict, Set, List

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

def generate_image_index(root_folder: str) -> Dict[str, str]:
    """Generate image base name index with full paths"""
    file_index = {}
    abs_root = pathlib.Path(root_folder).resolve()
    
    for file_path in abs_root.rglob('*'):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            uid = file_path.stem.split('_')[0].lower()
            file_index[uid] = str(file_path)
    
    glog.get_logger().info(f"Indexed {len(file_index)} images")
    return file_index

class TiRADSChecker:
    def __init__(self, spreadsheet_path: str, sheet_name: str, 
                 uid_column: str, tirads_column: str):
        self._validate_inputs(spreadsheet_path, sheet_name, 
                            [uid_column, tirads_column])
        
        self.df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name)
        self.uid_col = uid_column.lower()
        self.tirads_col = tirads_column.lower()
        
        self.tirads_map = self._create_tirads_mapping()
        glog.get_logger().info(f"Loaded {len(self.tirads_map)} TiRADS entries")

    def _validate_inputs(self, path: str, sheet: str, columns: List[str]):
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"TiRADSChecker Spreadsheet not found: {path}")
        if not columns or any(not c for c in columns):
            raise ValueError("TiRADSChecker Invalid column names provided")

    def _create_tirads_mapping(self) -> Dict[str, int]:
        return self.df.set_index(
            self.df[self.uid_col].str.lower()
        )[self.tirads_col].to_dict()

    def get_tirads_level(self, uid: str) -> int:
        try:
            level = self.tirads_map.get(uid.lower(), -1)
            if pd.isna(level):
                glog.get_logger().warning(f"NaN value found for UID: {uid}")
                return -1
            return int(level)  # Handle float-formatted integers
        except (ValueError, TypeError) as e:
            glog.get_logger().warning(f"Invalid TiRADS value for UID {uid}: {str(e)}")
            return -1
        except Exception as e:
            glog.get_logger().error(f"Unexpected error processing UID {uid}: {str(e)}")
            return -1

class BlockListChecker:
    def __init__(self, spreadsheet_path: str, sheet_name: str, uid_column: str):
        self._validate_inputs(spreadsheet_path, sheet_name)
        
        self.df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name)
        self.uid_col = uid_column.lower()
        self.blocked_uids = self._load_blocked_uids()
        glog.get_logger().info(f"Loaded {len(self.blocked_uids)} blocked UIDs")

    def _validate_inputs(self, path: str, sheet: str):
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"Block list not found: {path}")

    def _load_blocked_uids(self) -> Set[str]:
        return set(self.df[self.uid_col].str.lower().unique())

    def is_blocked(self, uid: str) -> bool:
        return uid.lower() in self.blocked_uids

def generate_dataset(output_file: str, image_index: Dict[str, str], 
                   tirads_checker: TiRADSChecker, block_checker: BlockListChecker,
                   target_counts: Dict[int, int]):
    counts = {k: 0 for k in target_counts}
    
    with open(output_file, 'w') as f:
        f.write("FilePath,TiRADS\n")
        
        for uid, file_path in image_index.items():
            file_path = pathlib.Path(file_path)
            imageBasename = file_path.name
            if block_checker.is_blocked(uid):
                glog.get_logger().info(f"Blocked UID: {uid}")
                continue
                
            tirads_level = tirads_checker.get_tirads_level(uid)
            glog.get_logger().info(f"UID: {uid},  {imageBasename}, TiRADS: {tirads_level}")
            if tirads_level == -1 or tirads_level not in target_counts:
                continue
                
            if counts[tirads_level] < target_counts[tirads_level]:
                f.write(f"{imageBasename},{tirads_level}\n")
                counts[tirads_level] += 1
                glog.get_logger().info(f"Added {uid} (TiRADS {tirads_level})")
                
    glog.get_logger().info("Dataset generation completed")
    glog.get_logger().info(f"Final counts: {counts}")

def main_generateTiradsDataset():
    parser = argparse.ArgumentParser(description='TiRADS Dataset Generator')
    parser.add_argument('image_root', help='Root directory containing medical images')
    parser.add_argument('img_info_sheet', help='Path to TiRADS spreadsheet')
    parser.add_argument('block_items_sheet', help='Path to block list spreadsheet')
    parser.add_argument('-o', '--output', default='tirads_datasetPatch6k.csv', 
                      help='Output CSV file path')
    args = parser.parse_args()

    try:
        # Initialize all components
        image_index = generate_image_index(args.image_root)
        tirads_checker = TiRADSChecker(args.img_info_sheet, 'origintable',
                                      'sop_uid', 'ti_rads')
        block_checker = BlockListChecker(args.block_items_sheet, 
                                        'verify_3000_tirads1_5', 'sop_uid')
        
        # Define target counts (example: adjust based on requirements)
        everyTypeCount=6000
        target_counts = {
            1: everyTypeCount,
            2: everyTypeCount,
            3: everyTypeCount,
            4: everyTypeCount,
            5: everyTypeCount,
        }
        alreadyAppendCount=[237, 1515, 3064, 6000, 6000]
        for itck, itcv in target_counts.items():
            target_counts[itck]=itcv-alreadyAppendCount[itck-1]
        glog.get_logger().info(f" targets counnts={target_counts}")
        
        generate_dataset(args.output, image_index, tirads_checker,
                        block_checker, target_counts)
        
    except Exception as e:
        glog.get_logger().error(f"Failed to generate dataset: {str(e)}")
        raise SystemExit(1) from e

if __name__ == "__main__":
    glog.glogger = glog.initLogger("genTirads_dataset.log")
    main_generateTiradsDataset()
