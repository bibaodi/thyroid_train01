import os
import sys
import argparse
from tqdm import tqdm
import pandas as pd
import shutil
from pathlib import Path
from typing import List, Dict, Set, Tuple

def check_disk_space(path, required_gb=10):
    """Check if there's enough disk space at the given path."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free // (2**30)  # Convert bytes to GB
    if free_gb < required_gb:
        print(f"Warning: Only {free_gb}GB of free space available (required: {required_gb}GB), total: {total // (2**20)}MB, used: {used // (2**20)}MB.")
        return False
    return True

class ImageSubsetCreator:
    """Class to create subsets of images based on spreadsheet criteria"""
    
    def __init__(self, image_folder: str, spreadsheet_file: str, output_folder: str, 
                 filter_values: List[str], column_name: str = 'type', sheet_name: str = None):
        """
        Initialize the ImageSubsetCreator.
        
        Args:
            image_folder: Path to the input image folder
            spreadsheet_file: Path to the input spreadsheet file
            output_folder: Path to the output subset folder
            filter_values: List of values to filter on
            column_name: Column name to compare filter values against
            sheet_name: Sheet name if spreadsheet has multiple sheets
        """
        self.image_folder = image_folder
        self.spreadsheet_file = spreadsheet_file
        self.output_folder = output_folder
        self.filter_values = filter_values
        self.column_name = column_name
        self.sheet_name = sheet_name
        self.statistics = {
            'matched_items': 0,
            'found_images': 0,
            'case_folders': set()
        }
    
    def validate_parameters(self) -> bool:
        """Validate all input parameters before processing."""
        # Check if image folder exists
        if not os.path.exists(self.image_folder):
            print(f"Error: Image folder '{self.image_folder}' does not exist.")
            return False
        
        # Check if spreadsheet file exists
        if not os.path.exists(self.spreadsheet_file):
            print(f"Error: Spreadsheet file '{self.spreadsheet_file}' does not exist.")
            return False
        
        # Check spreadsheet file extension
        ext = os.path.splitext(self.spreadsheet_file)[1].lower()
        if ext not in ['.csv', '.xlsx', '.xls']:
            print(f"Error: Unsupported spreadsheet format. Please provide a CSV or Excel file.")
            return False

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        if not check_disk_space(self.output_folder, required_gb=0.6):
            print("Not enough disk space available. Aborting.")
            return False
        
        return True
    
    def load_spreadsheet(self) -> pd.DataFrame:
        """Load the spreadsheet file with only necessary columns."""
        try:
            # Define the columns we actually need
            # We need 'access_no' and 'sop_uid' for image path construction
            # and the column specified by self.column_name for filtering
            required_columns = ['access_no', 'sop_uid', self.column_name]
            
            print(f"Loading spreadsheet file: {self.spreadsheet_file}")
            print(f"Only loading necessary columns: {', '.join(required_columns)}")
            
            # Load only the required columns
            if self.spreadsheet_file.endswith('.xlsx') or self.spreadsheet_file.endswith('.xls'):
                if self.sheet_name:
                    # For Excel files with specific sheet name
                    data = pd.read_excel(
                        self.spreadsheet_file,
                        sheet_name=self.sheet_name,
                        usecols=required_columns
                    )
                else:
                    # For Excel files with default sheet
                    data = pd.read_excel(
                        self.spreadsheet_file,
                        usecols=required_columns
                    )
            elif self.spreadsheet_file.endswith('.csv'):
                # For CSV files
                data = pd.read_csv(
                    self.spreadsheet_file,
                    usecols=required_columns
                )
            else:
                raise ValueError(f"Unsupported file format: {self.spreadsheet_file}")
            
            print(f"Successfully loaded {len(data)} rows from the spreadsheet.")
            return data
        except Exception as e:
            print(f"Error loading spreadsheet file: {str(e)}")
            raise
    
    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data based on column name and filter values."""
        # Check if column exists in the dataframe
        if self.column_name not in data.columns:
            print(f"Error: Column '{self.column_name}' not found in the spreadsheet.")
            sys.exit(1)
        
        # Filter data
        filtered_data = data[data[self.column_name].isin(self.filter_values)]
        self.statistics['matched_items'] = len(filtered_data)
        
        print(f"Found {len(filtered_data)}/{len(data)} items matching the filter criteria.")
        return filtered_data
    
    def get_case_folders(self) -> Dict[str, str]:
        """Get all case folders in the image directory."""
        case_folders = {}
        try:
            for item in os.listdir(self.image_folder):
                item_path = os.path.join(self.image_folder, item)
                if os.path.isdir(item_path):
                    case_folders[item] = item_path
        except Exception as e:
            print(f"Error accessing image folder: {str(e)}")
            sys.exit(1)
        
        return case_folders
    
    def prepare_image_list(self, filtered_data: pd.DataFrame) -> List[Dict[str, str]]:
        """Prepare list of images to copy based on filtered spreadsheet data."""
        image_list = []
        
        for _, row in filtered_data.iterrows():
            try:
                # Check if required columns exist in the row
                if 'access_no' not in row or 'sop_uid' not in row:
                    print(f"Warning: Missing required columns in row: {row}")
                    continue
                
                access_no = row['access_no']
                sop_uid = row['sop_uid']
                
                # Skip if access_no or sop_uid is NaN
                if pd.isna(access_no) or pd.isna(sop_uid):
                    continue
                
                # Build the image path using the provided method
                image_path = os.path.join(self.image_folder, str(access_no), f"{sop_uid}.jpg")
                
                # Check if the image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                # Get the filter value for this case
                filter_value = row[self.column_name]
                
                # Add to image list
                image_list.append({
                    'access_no': str(access_no),
                    'sop_uid': sop_uid,
                    'image_path': image_path,
                    'filter_value': filter_value
                })
            except Exception as e:
                print(f"Error preparing image info for row {row}: {str(e)}")
        
        print(f"Prepared {len(image_list)} valid image entries for copying.")
        return image_list

    def copy_matching_images(self, image_list: List[Dict[str, str]]) -> None:
        """Copy matching images to the output directory based on prepared image list."""
        
        # Create directories for each filter value
        filter_dirs = {}
        for image_info in image_list:
            filter_value = image_info['filter_value']
            if filter_value not in filter_dirs:
                filter_dir = os.path.join(self.output_folder, str(filter_value))
                os.makedirs(filter_dir, exist_ok=True)
                filter_dirs[filter_value] = filter_dir
        
        # Copy images based on prepared image list with progress bar
        print(f"Copying {len(image_list)} images...")
        for image_info in tqdm(image_list, desc="Copying files", unit="file"):
            try:
                access_no = image_info['access_no']
                sop_uid = image_info['sop_uid']
                source_path = image_info['image_path']
                filter_value = image_info['filter_value']
                
                # Create destination directory structure
                dest_case_dir = os.path.join(filter_dirs[filter_value], access_no)
                os.makedirs(dest_case_dir, exist_ok=True)
                
                # Define destination image path
                dest_image_path = os.path.join(dest_case_dir, f"{sop_uid}.jpg")
                
                # Copy the image file
                shutil.copy2(source_path, dest_image_path)
                
                # Update statistics
                self.statistics['found_images'] += 1
                self.statistics['case_folders'].add(access_no)
                
            except OSError as e:
                # Check if the error is related to disk space
                if e.errno == 28:  # Error code for 'No space left on device'
                    print(f"Critical error: {str(e)}")
                    print("Aborting operation due to insufficient disk space.")
                    # Print current statistics before exiting
                    self.print_statistics()
                    sys.exit(1)
                else:
                    print(f"Error copying image {image_info['image_path']}: {str(e)}")
            except Exception as e:
                # Handle other types of exceptions
                print(f"Unexpected error copying image {image_info['image_path']}: {str(e)}")

    def run(self) -> None:
        """Run the image subset creation process."""
        # Validate parameters
        if not self.validate_parameters():
            sys.exit(1)
        
        print(f"Starting image subset creation process...")
        print(f"\nParameters:")
        print(f"- Image folder: {self.image_folder}")
        print(f"- Spreadsheet file: {self.spreadsheet_file}")
        print(f"- Output folder: {self.output_folder}")
        print(f"- Filter values: {', '.join(self.filter_values)}")
        print(f"- Column name: {self.column_name}")
        if self.sheet_name:
            print(f"- Sheet name: {self.sheet_name}")
        
        # Load and filter data
        print(f"\nLoading and filtering spreadsheet data...")
        data = self.load_spreadsheet()
        filtered_data = self.filter_data(data)
        
        # Prepare list of images to copy
        print(f"\nPreparing list of images to copy...")
        image_list = self.prepare_image_list(filtered_data)
        
        # Copy matching images
        print(f"\nCopying images to output directory...")
        self.copy_matching_images(image_list)
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self) -> None:
        """Print statistics about the subset creation process."""
        print(f"\n\n===== Statistics =====")
        print(f"Total items matched in spreadsheet: {self.statistics['matched_items']}")
        print(f"Total images found and copied: {self.statistics['found_images']}")
        print(f"Total unique case folders processed: {len(self.statistics['case_folders'])}")
        print(f"Output directory: {self.output_folder}")
        print(f"====================")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    
    _usage="""
    (google_ai_edgePy311) eton@M2703:~/job-trainThyUS/71-datasets/251016-efficientNetDatas/srcs$ python image_subset_creator.py 
        --image-folder /home/eton/job-trainThyUS/71-datasets/251016-efficientNetDatas/10-datasets/dataset_images/3nodule_images   
        --spreadsheet /home/eton/job-trainThyUS/71-datasets/251016-efficientNetDatas/dataset/all_verify_sop_with_predictions.csv   
        --filter-values 0809v3   
        --output /tmp/subsets
    """
    parser = argparse.ArgumentParser(description='Create image subsets based on spreadsheet criteria', usage=_usage)
    
    parser.add_argument('--image-folder', '-i', required=True,
                        help='Path to the input image folder')
    
    parser.add_argument('--spreadsheet', '-s', required=True,
                        help='Path to the input spreadsheet file')
    
    parser.add_argument('--filter-values', '-f', required=True,
                        help='Filter values separated by commas, e.g., "a,b,c"')
    
    parser.add_argument('--column', '-c', default='type',
                        help='Column name to compare filter values against (default: "type")')
    
    parser.add_argument('--sheet', '-sh',
                        help='Sheet name if spreadsheet has multiple sheets')
    
    parser.add_argument('--output', '-o', required=True,
                        help='Path to the output subset folder')
    
    return parser.parse_args()


def main() -> None:
    """Main function to run the image subset creator."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Process filter values
    filter_values = [v.strip() for v in args.filter_values.split(',')]
    
    # Create and run the image subset creator
    creator = ImageSubsetCreator(
        image_folder=args.image_folder,
        spreadsheet_file=args.spreadsheet,
        output_folder=args.output,
        filter_values=filter_values,
        column_name=args.column,
        sheet_name=args.sheet
    )
    
    creator.run()


if __name__ == '__main__':
    main()