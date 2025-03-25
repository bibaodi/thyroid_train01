"""
File: SpreadSheetChecker.py
Author: eton.bi
Date: 2025-03-24
Version: 1.0
Description: First version of spreadsheet checking application.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpreadSheetChecker:
    def __init__(self, file_path: Union[str,Path], sheet_name: str, 
                 columns: list, primary_key: str):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.columns = columns
        self.primary_key = primary_key
        self.data = None
        self.error = False
        
        try:
            self._validate_inputs()
            self._load_data()
            logging.info("Spreadsheet loaded successfully")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            self.error = True

    def _validate_inputs(self) -> None:
        """Perform all safety checks before loading data"""
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not Path(self.file_path).is_file():
            raise ValueError(f"Not a file: {self.file_path}")
        if self.primary_key not in self.columns:
            raise ValueError("Primary key must be in columns list")

    def _load_data(self) -> None:
        """Load and validate spreadsheet data"""
        try:
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, 
                              usecols=self.columns + [self.primary_key])
            
            if self.primary_key not in df.columns:
                raise ValueError(f"Primary key column '{self.primary_key}' not found")
                
            self.data = df.set_index(self.primary_key)
        except Exception as e:
            raise RuntimeError(f"Data loading error: {str(e)}")

    def find_target_row(self, target_value: str) -> Optional[Dict[str, Union[str, int, float]]]:
        """Find row by primary key value, return None if not found"""
        if self.error:
            logging.warning("Instance in error state")
            return None
            
        if not target_value or not isinstance(target_value, str):
            logging.error("Invalid target value")
            return None
            
        try:
            row = self.data.loc[target_value].to_dict()
            return row
        except KeyError:
            return None
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return None

    def get_item_value(self, row_data: dict, column_name: str) -> Optional[Union[str, int, float]]:
        """Get specific column value from found row data"""
        if self.error:
            logging.warning("Instance in error state")
            return None
            
        if column_name not in self.columns:
            logging.error(f"Invalid column: {column_name}")
            return None
            
        return row_data.get(column_name)

def test():
    # Initialize checker
    checker = SpreadSheetChecker(
        file_path="2223multi_nodule_45434.xlsx",
        sheet_name="origintable",
        columns=["sop_uid", "ti_rads", "bethesda"],
        primary_key="sop_uid"
    )

    if not checker.error:
        # Search for target value
        row = checker.find_target_row("02.201705181524.01.0003.1495093244")
        
        if row:
            ti_rads = checker.get_item_value(row, "ti_rads")
            bethesda = checker.get_item_value(row, "bethesda")
            print(f"Found tirads: {ti_rads}, bethesda={bethesda}")
        else:
            print("Target not found")
    else:
        print("Initialization failed")

if "__main__" == __name__:
    test()