import os
import shutil
import pandas as pd

class DatasetOrganizer:
    def __init__(self, input_root, metadata_csv, output_root):
        self.input_root = input_root
        self.metadata_csv = metadata_csv
        self.output_root = output_root
        self.uid_map = self._read_csv_mapping()
    
    def _read_csv_mapping(self):
        """Read CSV into memory and create UID to bethesda mapping"""
        df = pd.read_csv(self.metadata_csv)
        return df.set_index('UID')['bethesda'].to_dict()
    
    def _get_target_dir(self, bethesda):
        """Determine target directory based on bethesda value"""
        if bethesda == 2:
            return os.path.join(self.output_root, '2benign')
        elif bethesda == 6:
            return os.path.join(self.output_root, '6Malign')
        return None
    
    def process_files(self):
        """Main processing method that walks through directories"""
        # Create output directories if they don't exist
        for dir_path in [os.path.join(self.output_root, d) for d in ('2benign', '6Malign')]:
            os.makedirs(dir_path, exist_ok=True)

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
    parser = argparse.ArgumentParser(description='Organize files based on Bethesda classification')
    parser.add_argument('-i', '--input_root', required=True, help='Root directory of input dataset')
    parser.add_argument('-m', '--metadata_csv', required=True, help='CSV file containing metadata')
    parser.add_argument('-o', '--output_root', required=True, help='Root directory for organized output')
    args = parser.parse_args()
    
    organizer = DatasetOrganizer(args.input_root, args.metadata_csv, args.output_root)
    organizer.process_files()

if __name__ == "__main__":
    main()
