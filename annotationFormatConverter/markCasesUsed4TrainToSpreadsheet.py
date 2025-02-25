"""
Author: [eton ]
Email: [bibaodi@jishimed.com]
Date: 250208
Version: 1.0
Description: This script maps cases from a directory to an Excel spreadsheet and marks them as used for training.
"""

import os
import pandas as pd
from datetime import datetime

class CaseMapper:
    """
    A class to map cases from a directory to an Excel spreadsheet and mark them as used for training.
    Attributes:
        directory (str): The directory containing case sub-folders.
        excel_file (str): The path to the Excel file.
        sheet_name (str): The name of the sheet in the Excel file to be processed.
        column_name (str): The column name in the Excel sheet to be matched with case names.
        suffix (str): The suffix to be added to the output file name.
        df (pd.DataFrame): The DataFrame containing the Excel sheet data.
        converted_cases (dict): A dictionary mapping converted case names to original case names.
        matched_items (list): A list of tuples containing matched original and converted case names.
    Methods:
        open_excel_file():
            Opens the Excel file and loads the specified sheet into a DataFrame.
        read_and_convert_cases():
            Reads the case names from the directory and converts them to the access_no format.
        reverse_getPath4AfterConverted(case_name):
            Reverses the logic of getPath4AfterConverted to obtain the original case name.
        compare_and_mark_rows():
            Compares the converted case names with the Excel sheet and marks the rows used for training.
        log_results(first_success, first_failure, total_cases, matched_count):
            Logs the results of the comparison, including the first successful and failed matches.
        save_results():
            Saves the updated DataFrame to a new Excel file and writes the matched items to a text file.
    """
    def __init__(self, directory, excel_file, sheet_name, column_name, suffix):
        self.directory = directory
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.column_name = column_name
        self.suffix = suffix
        self.df = None
        self.converted_cases = None
        self.matched_items = []

    def open_excel_file(self):
        # Check if the file exists
        if not os.path.exists(self.excel_file):
            raise FileNotFoundError(f"The file {self.excel_file} does not exist.")
        
        # Read the Excel file and select the target sheet
        self.df = pd.read_excel(self.excel_file, sheet_name=self.sheet_name)

    def read_and_convert_cases(self):
        # Check if the directory exists
        if not os.path.exists(self.directory):
            raise FileNotFoundError(f"The directory {self.directory} does not exist.")
        
        # Get the list of all case names (sub-folder names) in the parent folder
        cases = os.listdir(self.directory)
        
        # Convert case names to access_no format (reverse getPath4AfterConverted logic)
        self.converted_cases = {self.reverse_getPath4AfterConverted(case): case for case in cases}

    def reverse_getPath4AfterConverted(self, case_name):
        # Reverse the logic of getPath4AfterConverted
        if case_name.startswith("301PACS"):
            case_name = case_name.replace("301PACS", "", 1)
            
            if case_name.startswith("02"):
                case_name = case_name.replace("-", ".20", 1)
            elif case_name.startswith("22"):
                case_name = case_name.replace("-", ".000000", 1)
        
        return case_name

    def compare_and_mark_rows(self):
        self.df['used4train'] = 0
        first_success = None
        first_failure = None
        matched_count = 0
        
        for index, row in self.df.iterrows():
            if row[self.column_name] in self.converted_cases:
                self.df.at[index, 'used4train'] = 1
                matched_count += 1
                original_case = self.converted_cases[row[self.column_name]]
                self.matched_items.append((original_case, row[self.column_name]))
                if first_success is None:
                    first_success = row[self.column_name]
            else:
                if first_failure is None:
                    first_failure = row[self.column_name]
        
        return first_success, first_failure, matched_count

    def log_results(self, first_success, first_failure, total_cases, matched_count):
        if first_success:
            print(f"First successful match: {first_success}")
        if first_failure:
            print(f"First failed match: {first_failure}")
        
        print(f"Total number of cases: {total_cases}")
        print(f"Number of matched items: {matched_count}")

    def save_results(self):
        base_name, ext = os.path.splitext(self.excel_file)
        datetime_suffix = datetime.now().strftime("%y%m%d%H%M%S")
        new_excel_file = f"{base_name}_{self.suffix}_{datetime_suffix}{ext}"
        
        for attempt in range(3):
            try:
                self.df.to_excel(new_excel_file, sheet_name=self.sheet_name, index=False)
                print(f"Saved updated data to {new_excel_file} with used4train column based on cases in {self.directory}")
                break
            except Exception as e:
                print(f"Failed to save file {new_excel_file}: {e}")
                datetime_suffix = datetime.now().strftime("%y%m%d%H%M%S")
                new_excel_file = f"{base_name}_{self.suffix}_{datetime_suffix}_retry{attempt+1}{ext}"
        else:
            print("Failed to save the file after 3 attempts.")
        
        self.matched_items.sort()
        matched_file = f"{base_name}_matched_items.txt"
        with open(matched_file, 'w') as f:
            for original, converted in self.matched_items:
                f.write(f"{original} -> {converted}\n")
        print(f"Saved matched items to {matched_file}")

def main():
    directory = '/mnt/f/241129-xin1zhipu-thyroid-datas/17--labelmeFormatOrganized/301pacsDataInLbmfmtRangeY22-24'
    excel_file = os.path.join(directory, '301PACS_database_RangeY22-24_V250113.xlsx')
    sheet_name = 'origintable'
    column_name = 'access_no'
    suffix = 'updated'

    case_mapper = CaseMapper(directory, excel_file, sheet_name, column_name, suffix)
    case_mapper.open_excel_file()
    case_mapper.read_and_convert_cases()
    first_success, first_failure, matched_count = case_mapper.compare_and_mark_rows()
    case_mapper.log_results(first_success, first_failure, len(case_mapper.converted_cases), matched_count)
    case_mapper.save_results()

if __name__ == "__main__":
    main()