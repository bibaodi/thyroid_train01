import pandas as pd
import logging
import os
from datetime import datetime


def setup_logging():
    """Setup logging with filename and line number"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s:%(lineno)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def fill_empty_values(df):
    """
    Fill empty values in the dataframe with '其他'.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with empty values filled
    """
    # Replace empty strings, NaN, and None values with '其他'
    df_filled = df.fillna('其他')
    df_filled = df_filled.replace('', '其他')
    df_filled = df_filled.replace(r'^\s*$', '其他', regex=True)  # Replace whitespace-only cells
    return df_filled


def generate_new_spreadsheet(input_file_path, output_rows=15000):
    """
    Generate a new spreadsheet file from input file with specific transformations.
    
    Args:
        input_file_path (str): Path to input CSV file
        output_rows (int): Number of rows to include in output (default 15000)
    """
    logger = logging.getLogger(__name__)
    
    # Read the input file
    logger.info(f"Reading input file: {input_file_path}")
    df = pd.read_csv(input_file_path)
    
    # Select only the required columns
    selected_columns = ['bom', 'bethesda', 'position', 'composition', 'echo', 'margin', 'foci']
    df_selected = df[selected_columns].copy()
    
    # Limit to top 15000 rows
    df_selected = df_selected.head(output_rows)
    
    # Rename 'bom' column to '良性恶性'
    #df_selected.rename(columns={'bom': '良性恶性'}, inplace=True)
    
    # Change values in '良性恶性' column: 0 -> '良性', 1 -> '恶性'
    df_selected['bom'] = df_selected['bom'].map({0: '良性', 1: '恶性'})
    
    # Fill empty values with '其他'
    df_selected = fill_empty_values(df_selected)
    
    # Rename columns to Chinese labels
    column_mapping = {
        'case_number': '病例编号',
        'bom': '良性恶性',
        'bethesda': 'bethesda分级',
        'position': '结节位置',
        'composition': '结节成分',
        'echo': '结节回声性',
        'margin': '结节边缘',
        'foci': '结节回声强度'
    }
    
    # Add case number column starting from 1 to N
    df_selected.insert(0, 'case_number', range(1, len(df_selected) + 1))
    
    # Rename the columns using the mapping
    df_selected.rename(columns=column_mapping, inplace=True)
    
    # Generate output filename with datetime suffix
    input_dir = os.path.dirname(input_file_path)
    input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"out_{input_filename}_{timestamp}.csv"
    output_path = os.path.join(input_dir, output_filename)
    
    # Save the new dataframe to CSV
    logger.info(f"Saving output file: {output_path}")
    df_selected.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Successfully created {output_path} with {len(df_selected)} rows")
    return output_path


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Default input file path
    input_file = "/mnt/datas/42workspace/32-project_thyroAid/260127-register-files/trainDataFor-database-certificate2.9/all_matched_sop_v11.csv"
    
    # Allow user to input file path
    user_input = input(f"Enter input file path (or press Enter to use default: {input_file}): ").strip()
    if user_input:
        input_file = user_input
    
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return
    
    try:
        output_file = generate_new_spreadsheet(input_file)
        logger.info(f"Process completed successfully. Output file: {output_file}")
    except Exception as e:
        logger.error(f"Error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()