import os
import shutil
import pandas as pd

class DatasetOrganizer:
    CATEGORY_MAPPINGS = {
        'bethesda26': {2: '2Benign', 6: '6Malign'},
         'tirads15': {1: 'TR1', 2: 'TR2', 3: 'TR3', 4: 'TR4', 5: 'TR5'}
    }

    def __init__(self, input_root, metadata_csv, output_root, classify_category='bethesda26'):
        self.input_root = input_root
        self.metadata_csv = metadata_csv
        self.output_root = output_root
        self.classify_category = classify_category
        self.uid_map = self._read_csv_mapping()
        
        if classify_category not in self.CATEGORY_MAPPINGS:
            raise ValueError(f"Invalid classification category: {classify_category}. "
                             f"Available options: {list(self.CATEGORY_MAPPINGS.keys())}")

    def _isClsCategoryBethesda(self):
        return self.classify_category == 'bethesda26'
    
    def _isClsCategoryTIRADS(self):
        return self.classify_category == 'tirads15'

    def _read_csv_mapping(self):
        """Read CSV into memory and create UID to bethesda mapping"""
        df = pd.read_csv(self.metadata_csv)
        if self._isClsCategoryBethesda():
            return df.set_index('UID')['bethesda'].to_dict()
        elif self._isClsCategoryTIRADS():
            return df.set_index('ImageName')['TiRADS'].to_dict()
    def _get_target_dir(self, category_value):
        """Determine target directory based on category value"""
        category_map = self.CATEGORY_MAPPINGS[self.classify_category]
        return os.path.join(self.output_root, category_map.get(category_value))

    def process_files(self):
        """Main processing method that walks through directories"""
        # Create output directories based on current category mapping
        category_map = self.CATEGORY_MAPPINGS[self.classify_category]
        for dir_name in category_map.values():
            os.makedirs(os.path.join(self.output_root, dir_name), exist_ok=True)

        # Walk through all files in input directory
        for root, _, files in os.walk(self.input_root):
            for file in files:
                self._process_single_file(root, file)
    
    def _process_single_file(self, root, file):
        """Process individual file"""
        file_path = os.path.join(root, file)
        base_name = os.path.splitext(file)[0]
        #02.202312250193.01.21446.0005.08053200704_crop-enlarged-crop --> remove '-enlarged-crop'
        matchStr = base_name.split('-')[0] if '-' in base_name else base_name
        try:
            bethesda = self.uid_map[matchStr]
            if target_dir := self._get_target_dir(bethesda):
                shutil.copy2(file_path, os.path.join(target_dir, file))
        except KeyError:
            print(f"Warning: No CSV entry found for {base_name}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Organize files based on classification')
    parser.add_argument('-i', '--input_root', required=True, help='Root directory of input dataset')
    parser.add_argument('-m', '--metadata_csv', required=True, help='CSV file containing metadata')
    parser.add_argument('-o', '--output_root', required=True, help='Root directory for organized output')
    parser.add_argument('-c', '--classify_category', default='bethesda26',
                      choices=DatasetOrganizer.CATEGORY_MAPPINGS.keys(),
                      help='Classification category to use')
    args = parser.parse_args()
    
    organizer = DatasetOrganizer(args.input_root, args.metadata_csv, 
                               args.output_root, args.classify_category)
    organizer.process_files()

if __name__ == "__main__":
    main()
