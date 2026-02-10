import pandas as pd
import logging
import argparse
import re
import os
from datetime import datetime


def setup_logging():
    """Setup logging with filename and line number"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s:%(lineno)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def clean_instrument_model(value):
    """Remove prefix in parentheses from instrument model column"""
    if pd.isna(value) or value == '':
        return "PHILIPS TypeA"
    
    value_str = str(value)
    
    # Handle case where there's a closing parenthesis without opening one at the beginning
    # e.g., "外科1诊室)LOGIQ E9" - remove up to and including the first ')'
    if value_str.startswith(')'):
        # If it starts with ), remove that and continue processing
        value_str = value_str[1:]
    elif ')' in value_str and not value_str.startswith('('):
        # If there's a ) somewhere but doesn't start with (, 
        # remove everything up to and including the first )
        parts = value_str.split(')', 1)  # Split only at the first occurrence
        if len(parts) > 1:
            value_str = parts[1]  # Take the part after the first )
    
    # Remove anything in parentheses at the beginning
    cleaned_value = re.sub(r'^\([^)]*\)', '', value_str).strip()
    
    # If the result is empty, return default value
    if not cleaned_value:
        return "PHILIPS TypeA"
    
    return cleaned_value


def generate_dynamicdb_presentation(input_file_path, output_rows=1100):
    """
    Generate a presentation spreadsheet from input file with specific transformations.
    
    Args:
        input_file_path (str): Path to input Excel file
        output_rows (int): Number of rows to include in output (default 1100)
    """
    logger = logging.getLogger(__name__)
    
    # Read the input file
    logger.info(f"Reading input file: {input_file_path}")
    df = pd.read_excel(input_file_path)
    
    # Select only the required columns
    selected_columns = ['性别_ultrasound', '年龄','录像', '工作站', '仪器型号', '检查项目']
    df_selected = df[selected_columns].copy()
    df_selected.rename(columns={'性别_ultrasound': '性别'}, inplace=True)
    df_selected.rename(columns={'录像': '动态数据'}, inplace=True)
    
    # Apply filters
    # Filter 1: Clean instrument model column by removing prefix in parentheses
    df_selected['仪器型号'] = df_selected['仪器型号'].apply(clean_instrument_model)
    
    # Filter 2: Keep only rows where '检查项目' contains '甲状腺'
    #df_selected = df_selected[df_selected['检查项目'].str.contains('甲状腺', na=False)]
    df_selected = df_selected[~df_selected['检查项目'].str.contains('颈动脉', na=False)]
    
    # Limit to top 1100 rows
    df_selected = df_selected.head(output_rows)
    
    # Add case number column starting from 1 to N
    df_selected.insert(0, '病例编号', range(1, len(df_selected) + 1))
    
    # Generate output filename with datetime suffix in yymmddThhmmss format
    input_dir = os.path.dirname(input_file_path)
    input_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    timestamp = datetime.now().strftime("%y%m%dT%H%M%S")
    output_filename = f"outDynData_{input_filename}_{timestamp}.xlsx"
    output_path = os.path.join(input_dir, output_filename)
    
    # Save the new dataframe to Excel
    logger.info(f"Saving output file: {output_path}")
    df_selected.to_excel(output_path, index=False)
    
    logger.info(f"Successfully created {output_path} with {len(df_selected)} rows")
    return output_path


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Generate dynamic DB presentation from input file")
    parser.add_argument("input_file", help="Input Excel file path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        return
    
    try:
        output_file = generate_dynamicdb_presentation(args.input_file)
        logger.info(f"Process completed successfully. Output file: {output_file}")
    except Exception as e:
        logger.error(f"Error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()