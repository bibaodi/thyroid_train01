#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Echogenicity Dataset Generator v0.2
Author: Eton
Created: 250423
Filename: echogenicity_dataset_generator.py
Description: Generates medical image datasets with echogenicity labels
"""

import argparse
import pathlib
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
import glog
from typing import Dict, Set, List

# Third-party imports
sys.path.append(str(pathlib.Path(__file__).parent.parent/'utils'))
import glog

# --- Constants ---
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
EchoGenicityNameMapper = {
    u'等回声': 'ISOECHO',
    u'高回声': 'HPRECHO',
    u'低回声': 'HPOECHO',
    u'极低回声': 'MHYECHO',
    u'实性': 'SOLIDECHO',
    u'囊实性': 'CYSTICSOLID',
    u'囊性': 'CYSTICECHO',
    u'海绵样': 'SPONGIFORM',
    u'不清': 'MARGILLDEFINED',
    u'光滑': 'MARGCIRCUMSCRIBED',
    u'不规则': 'MARGIRREGULAR',
    u'外侵': 'MARGEXTRATHYR',
    u'点状强回声': 'FOCI_PUNCTATEECHOGENICITY',
    u'粗大钙化': 'FOCI_MACROCALCIFICATION',
    u'粗大钙化,点状强回声': 'FOCI_MACROCALCIFICATION',
    u'粗大钙化,边缘钙化': 'FOCI_MACROCALCIFICATION',
    u'边缘钙化': 'FOCI_PERIPHERALCALCIFICATION',
    U'边缘钙化,点状强回声': 'FOCI_PERIPHERALCALCIFICATION',
    u'nan': 'FOCI_NOTEXIST',
    float('nan'): 'FOCI_NOTEXIST',
}

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

class ImageLabelChecker:
    def __init__(self, spreadsheet_path: str, sheet_name: str, 
                 dataID_column: str, dataLabel_column: str):
        self._validate_inputs(spreadsheet_path, sheet_name, 
                            [dataID_column, dataLabel_column])
        
        self.m_df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name)
        self.m_dataid_col = dataID_column.lower()
        self.m_datalabel_col = dataLabel_column.lower()
        
        self.m_idLabel_map = self._create_dataIdLabel_mapping()
        self.m_echoGenicityNameMapper = EchoGenicityNameMapper
        glog.get_logger().info(f"Loaded {len(self.m_idLabel_map)} Data entries")

    def _validate_inputs(self, path: str, sheet: str, columns: List[str]):
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"ImageLabelChecker Spreadsheet not found: {path}")
        if not columns or any(not c for c in columns):
            raise ValueError("ImageLabelChecker Invalid column names provided")

    def _create_dataIdLabel_mapping(self) -> Dict[str, str]:
        return self.m_df.set_index(
            self.m_df[self.m_dataid_col].str.lower()
        )[self.m_datalabel_col].to_dict()

    def _labelName_to_abbreviation(self, label_name: str) -> str:
        """Convert label name to abbreviation"""
        return self.m_echoGenicityNameMapper.get(label_name, 'NotFound')

    def get_label_value(self, uid: str, useRawValue=False) -> str:
        """Get the label value for a given ID"""
        notfound = "notfound"
        dictKeyNotFound = "UIDnotfound"
        try:
            label_value = self.m_idLabel_map.get(uid.lower(), dictKeyNotFound)
            if (isinstance(notfound, str) and dictKeyNotFound == label_value):
                glog.get_logger().warning(f"NotFound UID: {uid}")
                return notfound
            if useRawValue:
                return label_value
            else:
                if (isinstance(label_value, float) and pd.isna(label_value)):
                    glog.get_logger().warning(f"NaN/nan value found for UID: {uid}")
                    label_value = str(label_value)
                return self._labelName_to_abbreviation(label_value)  # Handle float-formatted integers
        except (ValueError, TypeError) as e:
            glog.get_logger().warning(f"Invalid DataLabel value for UID {uid}: {str(e)}")
            return notfound
        except Exception as e:
            glog.get_logger().error(f"Unexpected error processing UID {uid}: {str(e)}")
            return notfound

class BlockListChecker:
    def __init__(self, spreadsheet_path: str, sheet_name: str, dataID_column: str):
        self.m_df = None
        self.m_dataid_col = None
        self.blocked_uids = []
        if self._validate_inputs(spreadsheet_path, sheet_name):
            self.m_df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name)
            self.m_dataid_col = dataID_column.lower()
            self.blocked_uids = self._load_blocked_uids()
        glog.get_logger().info(f"Loaded {len(self.blocked_uids)} blocked UIDs")

    def _validate_inputs(self, path: str, sheet: str):
        if not pathlib.Path(path).exists():
            glog.get_logger().error(FileNotFoundError(f"Block list not found: {path}"))
            return False
        return True

    def _load_blocked_uids(self) -> Set[str]:
        return set(self.m_df[self.m_dataid_col].str.lower().unique())

    def is_blocked(self, uid: str) -> bool:
        """Check if a UID is in the blocked list"""
        return uid.lower() in self.blocked_uids

def generate_dataset(
    output_file: str,
    image_index: Dict[str, str],
    tirads_checker: ImageLabelChecker,
    block_checker: BlockListChecker,
    target_counts: Dict[int, int]
):
    counts = {k: 0 for k in target_counts}
    
    with open(output_file, 'w') as f:
        f.write("ImageName,DataLabel\n")
        
        for uid, file_path in image_index.items():
            file_path = pathlib.Path(file_path)
            imageBasename = file_path.name
            if block_checker.is_blocked(uid):
                glog.get_logger().info(f"Blocked UID: {uid}")
                continue
                
            item_label = tirads_checker.get_label_value(uid)
            glog.get_logger().info(f"UID: {uid}, {imageBasename}, Label: {item_label}")
            if item_label not in target_counts:
                continue
                
            if counts[item_label] < target_counts[item_label]:
                f.write(f"{imageBasename},{item_label}\n")
                counts[item_label] += 1
                glog.get_logger().info(f"Added {uid} (DataLabel {item_label})")
    
    glog.get_logger().info("Dataset generation completed.")
    glog.get_logger().info(f"Final counts: {counts}")

def validate_input_parameters(args: argparse.Namespace):
    """Validate all input paths before main processing"""
    image_root = pathlib.Path(args.image_root)
    if not image_root.exists() or not image_root.is_dir():
        glog.get_logger().error(f"Image root directory not found: {args.image_root}")
        raise SystemExit(1)
    
    spreadsheet_path = pathlib.Path(args.img_info_sheet)
    if not spreadsheet_path.exists() or not spreadsheet_path.is_file():
        glog.get_logger().error(f"Input spreadsheet not found: {args.img_info_sheet}")
        raise SystemExit(1)

    blocklist_path = pathlib.Path(args.block_items_sheet)
    if not blocklist_path.exists() or not blocklist_path.is_file():
        glog.get_logger().error(f"Block list spreadsheet not found: {args.block_items_sheet}")
    return True 

def main_generateTiradsDataset():
    parser = argparse.ArgumentParser(description='DataLabel Dataset Generator')
    # Changed positional arguments to required keyword arguments
    parser.add_argument('-i', '--image-root', required=True, help='Root directory containing medical images')
    parser.add_argument('-s', '--img-info-sheet', required=True, help='Path to DataLabel spreadsheet')
    parser.add_argument('-b', '--block-items-sheet', required=True, help='Path to block list spreadsheet')
    parser.add_argument('-o', '--output', default='echoComposition_v01.250423.csv', 
                      help='Output CSV file path')
    args = parser.parse_args()

    # Validate inputs before main logic
    validate_input_parameters(args)
    
    # Create output directory if needed (kept here as it's output-related)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize all components
        image_index = generate_image_index(args.image_root)
        tirads_checker = ImageLabelChecker(args.img_info_sheet, 'sop_0422',
                                      'sop_uid', 'std_foci')
        block_checker = BlockListChecker(args.block_items_sheet, 
                                        'verify_3000_tirads1_5', 'sop_uid')
        
        # Define target counts (example: adjust based on requirements)
        everyTypeCount = 10

        # Convert to the new name mapping
        target_counts = {
            'FOCI_PUNCTATEECHOGENICITY': everyTypeCount,
            'FOCI_MACROCALCIFICATION': everyTypeCount,
            'FOCI_PERIPHERALCALCIFICATION': everyTypeCount,
            'FOCI_NOTEXIST': everyTypeCount,
        }
        label_keys = list(target_counts.keys())
        alreadyAppendCount=[0,0,0,0,0]
        for itck, itcv in target_counts.items():
            itemIdx = label_keys.index(itck)
            target_counts[itck]=itcv-alreadyAppendCount[itemIdx]
        glog.get_logger().info(f" targets counnts={target_counts}")
        
        generate_dataset(args.output, image_index, tirads_checker,
                        block_checker, target_counts)
        
    except Exception as e:
        glog.get_logger().error(f"Failed to generate dataset: {str(e)}")
        raise SystemExit(1) from e

if __name__ == "__main__":
    glog.glogger = glog.initLogger("echoComposition4_dataset.log")
    main_generateTiradsDataset()
