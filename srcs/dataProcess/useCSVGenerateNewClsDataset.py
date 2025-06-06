# -*- coding: utf-8 -*-
"""
File: useCSVGenerateNewClsDataset.py
Author: eton
Created: 250329
Description: Organize medical images based on classification metadata from CSV
"""

import os
import shutil
import pandas as pd

class DatasetOrganizer:
    CATEGORY_MAPPINGS = {
        'bethesda26': {2: '2Benign', 6: '6Malign'},
        'tirads15': {1: 'TR1', 2: 'TR2', 3: 'TR3', 4: 'TR4', 5: 'TR5'},
        'echoGenicity':{'ISOECHO': '0ISOECHO', 'HPRECHO': '1HPRECHO', 'HPOECHO': '2HPOECHO', 'MHYECHO': '3MHYECHO'},
        'echoComposition':{'SOLIDECHO': '0SOLID', 'CYSTICSOLID': '1CYSOL', 'CYSTICECHO': '2CYSTIC', 'SPONGIFORM': '3SPONGI'},
        'echoNoduleMargin':{'MARGILLDEFINED': '0MGILLD', 'MARGCIRCUMSCRIBED': '1MGCIRCU', 'MARGIRREGULAR': '2MGIRRE', 'MARGEXTRATHYR': '3MGEXTR'},
        'echoFoci':{'FOCI_PUNCTATEECHOGENICITY': '0FCPUNC', 'FOCI_MACROCALCIFICATION': '1FCMACA', 'FOCI_PERIPHERALCALCIFICATION': '2FCPPCA',},
    }

    def __init__(self, input_root, metadata_csv, output_root, classify_category='bethesda26'):
        self.m_input_root = input_root
        self.m_metadata_csv = metadata_csv
        self.m_output_root = output_root
        self.m_classify_category = classify_category
        self.m_uid_map = self._read_csv_mapping()
        
        if self.m_classify_category not in self.CATEGORY_MAPPINGS:
            raise ValueError(f"Invalid classification category: {self.m_classify_category}. "
                             f"Available options: {list(self.CATEGORY_MAPPINGS.keys())}")

    def _isClsCategoryBethesda(self):
        return self.m_classify_category == 'bethesda26'
    
    def _isClsCategoryTIRADS(self):
        return self.m_classify_category == 'tirads15'
    
    def _isClsCategoryEchoGenicity(self):
        return self.m_classify_category == 'echoGenicity'
    
    def _isClsCategoryEchoComposition(self):
        return self.m_classify_category == 'echoComposition'
    
    def _isClsCategoryEchoNoduleMargin(self):
        return self.m_classify_category == 'echoNoduleMargin'
    
    def _isClsCategoryEchoFoci(self):
        return self.m_classify_category == 'echoFoci'

    def _read_csv_mapping(self):
        """Read CSV into memory and create UID to bethesda mapping"""
        df = pd.read_csv(self.m_metadata_csv)
        if self._isClsCategoryBethesda():
            # Check and handle UID duplicates
            dup_count = df.duplicated(subset=['UID']).sum()
            if dup_count > 0:
                print(f"\nFound {dup_count} duplicate UIDs. Keeping first occurrence.")
                df = df.drop_duplicates(subset=['UID'], keep='first')
            
            bethesda_counts = df['bethesda'].value_counts().sort_index()
            print("\nBethesda Category Counts:")
            for value, count in bethesda_counts.items():
                print(f"Bethesda {int(value)}: {count} cases")
            print(f"Unique Cases: {len(df)}\n")
            
            return df.set_index('UID')['bethesda'].to_dict()
        elif self._isClsCategoryTIRADS():
            # Check and handle ImageName duplicates
            dup_count = df.duplicated(subset=['ImageName']).sum()
            if dup_count > 0:
                print(f"\nFound {dup_count} duplicate ImageNames. Keeping first occurrence.")
                df = df.drop_duplicates(subset=['ImageName'], keep='first')
            
            tirads_counts = df['TiRADS'].value_counts().sort_index()
            print("\nTiRADS Category Counts:")
            for value, count in tirads_counts.items():
                print(f"TiRADS {int(value)}: {count} cases")
            print(f"Unique Cases: {len(df)}\n")
            
            return df.set_index('ImageName')['TiRADS'].to_dict()
        elif self._isClsCategoryEchoGenicity() or self._isClsCategoryEchoComposition() or self._isClsCategoryEchoNoduleMargin() or self._isClsCategoryEchoFoci():
            # Check and handle ImageName duplicates
            dup_count = df.duplicated(subset=['ImageName']).sum()
            if dup_count > 0:
                print(f"\nFound {dup_count} duplicate ImageNames. Keeping first occurrence.")
                df = df.drop_duplicates(subset=['ImageName'], keep='first')
            
            dataLabel_counts = df['DataLabel'].value_counts().sort_index()
            print("\nDataLabel Category Counts:")
            for value, count in dataLabel_counts.items():
                print(f"DataLabel [{value}]: {count} cases")
            print(f"Unique Cases: {len(df)}\n")
            
            return df.set_index('ImageName')['DataLabel'].to_dict()

    def _get_target_dir(self, category_value):
        """Determine target directory based on category value"""
        category_map = self.CATEGORY_MAPPINGS[self.m_classify_category]
        return os.path.join(self.m_output_root, category_map.get(category_value))

    def process_files(self):
        """Main processing method that walks through directories"""
        # Create output directories based on current category mapping
        category_map = self.CATEGORY_MAPPINGS[self.m_classify_category]
        for dir_name in category_map.values():
            os.makedirs(os.path.join(self.m_output_root, dir_name), exist_ok=True)

        # Walk through all files in input directory
        for root, _, files in os.walk(self.m_input_root):
            for file in files:
                self._process_single_file(root, file)
    
    def _process_single_file(self, root, file):
        """Process individual file"""
        file_path = os.path.join(root, file)
        base_name = os.path.splitext(file)[0]
        #02.202312250193.01.21446.0005.08053200704_crop-enlarged-crop --> remove '-enlarged-crop'
        matchStr = base_name.split('-')[0] if '-' in base_name else base_name
        try:
            itemCls = self.m_uid_map[matchStr]
            if target_dir := self._get_target_dir(itemCls):
                shutil.copy2(file_path, os.path.join(target_dir, file))
        except KeyError:
            print(f"Warning: No CSV entry found for basename:{matchStr}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Organize files based on classification')
    parser.add_argument('-i', '--input_root', required=True, help='Root directory of input dataset')
    parser.add_argument('-m', '--metadata_csv', required=True, help='CSV file containing metadata')
    parser.add_argument('-o', '--output_root', required=True, help='Root directory for organized output')
    parser.add_argument('-c', '--classify_category', default='echoGenicity',
                      choices=DatasetOrganizer.CATEGORY_MAPPINGS.keys(),
                      help='Classification category to use')
    args = parser.parse_args()
    
    organizer = DatasetOrganizer(args.input_root, args.metadata_csv, 
                               args.output_root, args.classify_category)
    organizer.process_files()

if __name__ == "__main__":
    main()
