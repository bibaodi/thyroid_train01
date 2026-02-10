#!/usr/bin/env python3
"""
Application to match pathology and ultrasound spreadsheet files based on patient IDs.
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import sys
from typing import List, Tuple
from datetime import datetime

g_columnn_name_hasVideo = '录像'  # Global variable for the video column name, can be adjusted if needed
def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('match_spreadsheets.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_spreadsheet(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load a spreadsheet file (Excel or ODS) and return a DataFrame.
    
    Args:
        file_path: Path to the spreadsheet file
        sheet_name: Name of the sheet to load (optional)
    
    Returns:
        DataFrame containing the spreadsheet data
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.xlsx' or file_path.suffix.lower() == '.xls':
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        else:
            df = pd.read_excel(file_path, header=None)
    elif file_path.suffix.lower() == '.ods':
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='odf', header=None)
        else:
            df = pd.read_excel(file_path, engine='odf', header=None)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Check if we need to adjust headers
    # If the first row contains common Chinese medical record column names, treat it as header
    if len(df) > 0:
        first_row_values = df.iloc[0].values
        # Check if the first row contains common column names
        has_patient_id = any(('病人' in str(val) or 'patient' in str(val).lower() or 'exam' in str(val).lower()) and ('编号' in str(val) or 'ID' in str(val) or '号' in str(val)) for val in first_row_values)
        
        if has_patient_id:
            # First row has column names
            df.columns = df.iloc[0]  # Set first row as header
            df = df.drop(df.index[0]).reset_index(drop=True)  # Remove first row
        elif len(df) > 1:
            # Check if first row is metadata like "导出日期" and second row has column names
            first_cell = str(df.iloc[0, 0])
            second_row_first_val = str(df.iloc[1, 0])
            
            if '导出日期' in first_cell and ('病人' in second_row_first_val and ('编号' in second_row_first_val or 'ID' in second_row_first_val)):
                # Second row has column names
                df.columns = df.iloc[1]  # Set second row as header
                df = df.drop(df.index[0:2]).reset_index(drop=True)  # Remove first two rows
    
    return df


def match_dataframes(
    pathology_df: pd.DataFrame, 
    ultrasound_df: pd.DataFrame, 
    pathology_key_col: str, 
    ultrasound_key_col: str
) -> Tuple[pd.DataFrame, int]:
    """
    Match ultrasound data with pathology data based on patient IDs.
    
    Args:
        pathology_df: DataFrame containing pathology data
        ultrasound_df: DataFrame containing ultrasound data
        pathology_key_col: Column name for pathology patient ID
        ultrasound_key_col: Column name for ultrasound patient ID
    
    Returns:
        Tuple of (matched DataFrame, number of filtered items)
    """
    logger = logging.getLogger(__name__)
    
    # Check if required columns exist
    if pathology_key_col not in pathology_df.columns:
        raise KeyError(f"Column '{pathology_key_col}' not found in pathology data. Available columns: {list(pathology_df.columns)}")
    
    if ultrasound_key_col not in ultrasound_df.columns:
        raise KeyError(f"Column '{ultrasound_key_col}' not found in ultrasound data. Available columns: {list(ultrasound_df.columns)}")
    
    logger.info(f"Pathology data shape: {pathology_df.shape}")
    logger.info(f"Ultrasound data shape: {ultrasound_df.shape}")
    
    # Convert patient ID columns to string and strip whitespace for comparison
    pathology_df[pathology_key_col] = pathology_df[pathology_key_col].astype(str).str.strip()
    ultrasound_df[ultrasound_key_col] = ultrasound_df[ultrasound_key_col].astype(str).str.strip()
    
    # Count unique patient IDs before merge to calculate statistics properly
    unique_ultrasound_patients = set(ultrasound_df[ultrasound_key_col].unique())
    unique_pathology_patients = set(pathology_df[pathology_key_col].unique())
    
    # Find matching patients
    matching_patients = unique_ultrasound_patients.intersection(unique_pathology_patients)
    non_matching_patients = unique_ultrasound_patients - unique_pathology_patients
    
    logger.info(f"Unique ultrasound patients: {len(unique_ultrasound_patients)}")
    logger.info(f"Unique pathology patients: {len(unique_pathology_patients)}")
    logger.info(f"Matching patients: {len(matching_patients)}")
    logger.info(f"Non-matching ultrasound patients: {len(non_matching_patients)}")
    
    # Select only required columns from ultrasound data
    required_ultrasound_cols = ['病人ID', '姓名', '性别', '年龄', '录像', '检查项目', '仪器型号', '工作站', '超声所见', '超声提示']
    available_ultrasound_cols = [col for col in required_ultrasound_cols if col in ultrasound_df.columns]
    ultrasound_filtered_df = ultrasound_df[available_ultrasound_cols]
    
    # Rename '结果' column in pathology data to 'pathology-result' before merge if it exists
    pathology_col_name = '结果'
    if pathology_col_name in pathology_df.columns:
        pathology_df_renamed = pathology_df.rename(columns={pathology_col_name: 'pathology-result'})
    else:
        pathology_df_renamed = pathology_df
    
    # Perform inner join to keep only rows where ultrasound '病人ID' has matching value in pathology '病人编号'
    merged_df = pd.merge(
        ultrasound_filtered_df,
        pathology_df_renamed,
        left_on=ultrasound_key_col,
        right_on=pathology_key_col,
        how='inner',
        suffixes=('_ultrasound', '_pathology')
    )
    
    # Calculate filtered count based on unique patients that have no matches
    # Since each ultrasound patient could match multiple pathology records, resulting in duplication
    # We count how many unique ultrasound patients had no matches
    total_unique_ultrasound_patients = len(unique_ultrasound_patients)
    matched_unique_patients = len(matching_patients)
    filtered_count = len(non_matching_patients)
    
    logger.info(f"Successfully matched {matched_unique_patients}/{total_unique_ultrasound_patients} unique ultrasound patients with pathology data")
    logger.info(f"Filtered out {filtered_count} unique ultrasound patients (no matching pathology data)")
    
    return merged_df, filtered_count


def apply_customer_filters(df: pd.DataFrame, filters: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    Apply filters to a DataFrame based on column-value conditions.
    
    Args:
        df: DataFrame to apply filters to
        filters: List of tuples (column_name, operator, value) where:
                 - column_name: name of the column to filter on
                 - operator: 'eq' for equality, 'neq' for not equal, 'contains' for substring match, 'not_contains' for exclusion
                 - value: the value to compare against
    
    Returns:
        Filtered DataFrame
    """
    logger = logging.getLogger(__name__)
    original_length = len(df)
    
    for column_name, operator, value in filters:
        if column_name not in df.columns:
            logger.warning(f"Column '{column_name}' not found in DataFrame. Skipping filter.")
            continue
            
        if operator == 'eq':
            # Equality filter
            mask = df[column_name].astype(str).str.strip() == value
        elif operator == 'neq':
            # Not equal filter
            mask = df[column_name].astype(str).str.strip() != value
        elif operator == 'contains':
            # Contains filter
            mask = df[column_name].astype(str).str.contains(value, na=False)
        elif operator == 'not_contains':
            # Does not contain filter
            mask = ~df[column_name].astype(str).str.contains(value, na=False)
        else:
            logger.warning(f"Unknown operator '{operator}'. Supported operators: 'eq', 'neq', 'contains', 'not_contains'. Skipping filter.")
            continue
        
        df = df[mask]
        filtered_count = original_length - len(df)
        logger.info(f"Applied filter: [{column_name} {operator} '{value}'], total {original_length},filtered out {filtered_count},remaining {len(df)} records")
    
    return df


def combine_files(file_paths: List[str], file_type: str) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Load and combine multiple spreadsheet files into a single DataFrame.
    
    Args:
        file_paths: List of file paths to load
        file_type: Type of files being loaded (for logging purposes)
    
    Returns:
        Tuple of (combined DataFrame, list of individual DataFrames)
    """
    logger = logging.getLogger(__name__)
    
    dfs = []
    for file_path in file_paths:
        logger.info(f"Loading {file_type} file: {file_path}")
        df = load_spreadsheet(file_path)
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No {file_type} files provided")
    
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {file_type} data shape: {combined_df.shape}")
    
    return combined_df, dfs


def apply_data_filters(df: pd.DataFrame, custom_filters: List[Tuple[str, str, str]], filter_by_hasvideo: bool) -> Tuple[pd.DataFrame, int]:
    """
    Apply various filters to the DataFrame.
    
    Args:
        df: DataFrame to apply filters to
        custom_filters: List of custom filters to apply
        filter_by_hasvideo: Whether to apply video filter
    
    Returns:
        Tuple of (filtered DataFrame, number of records filtered by video)
    """
    logger = logging.getLogger(__name__)
    
    original_length = len(df)
    
    # Apply custom filters if provided
    if custom_filters:
        df = apply_customer_filters(df, custom_filters)
    
    # Apply video filter if enabled
    last_remaining_count = original_length
    if filter_by_hasvideo:
        if g_columnn_name_hasVideo in df.columns:
            logger.info(f"Applying {g_columnn_name_hasVideo} filter - keeping only records where {g_columnn_name_hasVideo} is '有'")
            before_filter_hasvideo_count = len(df)
            df = df[df[g_columnn_name_hasVideo].astype(str).str.strip() == '有']
            
            last_remaining_count = len(df)
            filtered_out_count = before_filter_hasvideo_count - last_remaining_count
            logger.info(f"after filter by hasvideo total {before_filter_hasvideo_count}, Filtered out {filtered_out_count}, remaining {last_remaining_count} records due to [{g_columnn_name_hasVideo} != '有']")
        else:
            logger.warning(f"{g_columnn_name_hasVideo} column not found in ultrasound data. Skipping video filter.")
    
    return df, (original_length - last_remaining_count)


def print_statistics(stats: dict):
    """
    Print statistics about the processed data.
    
    Args:
        stats: Dictionary containing statistics to print
    """
    print("\n--- Statistics ---")
    print(f"- items in pathology: {stats['pathology_items']} (from {stats['unique_pathology_patients']} unique patients)")
    
    if stats['filtered_out_by_condition'] > 0:
        print(f"- items in ultrasound: {stats['ultrasound_items_original']} -> {stats['ultrasound_items_after_filter']} after {g_columnn_name_hasVideo} filter (from {stats['unique_ultrasound_patients_original']} -> {stats['unique_ultrasound_patients_after_filter']} unique patients)")
        print(f"- total removed: {stats['filtered_out_by_condition']} items")
    else:
        print(f"- items in ultrasound: {stats['ultrasound_items_after_filter']} (from {stats['unique_ultrasound_patients_after_filter']} unique patients)")
    
    print(f"- Filtered out items: {stats['filtered_items']} unique patients (no matching pathology data)")
    print(f"- Matched unique patients: {stats['matching_unique_patients']}")
    print(f"- Final items in output: {stats['final_items']}")


def generate_statistics(combined_pathology_df: pd.DataFrame, combined_ultrasound_df: pd.DataFrame, 
                       original_ultrasound_count: int, ultrasound_dfs: List[pd.DataFrame], 
                       pathology_key_col: str, ultrasound_key_col: str, filtered_count: int, 
                       final_df: pd.DataFrame, filter_by_hasvideo: bool, filteredOut_by_condition: int) -> dict:
    """
    Generate statistics for the processed data.
    
    Args:
        combined_pathology_df: Combined pathology DataFrame
        combined_ultrasound_df: Combined and filtered ultrasound DataFrame
        original_ultrasound_count: Original count of ultrasound records
        ultrasound_dfs: List of original ultrasound DataFrames
        pathology_key_col: Column name for pathology patient ID
        ultrasound_key_col: Column name for ultrasound patient ID
        filtered_count: Number of items filtered during matching
        final_df: Final merged DataFrame
        filter_by_hasvideo: Whether video filter was applied
        filtered_by_video: Number of records filtered by video filter
    
    Returns:
        Dictionary containing statistics
    """
    logger = logging.getLogger(__name__)
    
    # Calculate statistics
    unique_pathology_patients = set(combined_pathology_df[pathology_key_col].unique())
    unique_ultrasound_patients = set(combined_ultrasound_df[ultrasound_key_col].unique())
    matching_patients = unique_ultrasound_patients.intersection(unique_pathology_patients)
    
    stats = {
        'pathology_items': len(combined_pathology_df),
        'ultrasound_items_original': original_ultrasound_count,
        'ultrasound_items_after_filter': len(combined_ultrasound_df),
        'filtered_out_by_condition': filteredOut_by_condition,
        'unique_pathology_patients': len(unique_pathology_patients),
        'unique_ultrasound_patients_original': len(set(pd.concat(ultrasound_dfs, ignore_index=True)[ultrasound_key_col].unique())) if filter_by_hasvideo else len(set(combined_ultrasound_df[ultrasound_key_col].unique())),
        'unique_ultrasound_patients_after_filter': len(set(combined_ultrasound_df[ultrasound_key_col].unique())) if len(combined_ultrasound_df) > 0 else 0,
        'matching_unique_patients': len(matching_patients),
        'filtered_items': filtered_count,  # Number of unique ultrasound patients with no matches
        'final_items': len(final_df)
    }
    
    return stats


def process_files(
    pathology_files: List[str], 
    ultrasound_files: List[str], 
    pathology_key_col: str, 
    ultrasound_key_col: str,
    filter_by_hasvideo: bool = False,
    custom_filters: List[Tuple[str, str, str]] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Process all pathology and ultrasound files and merge them.
    
    Args:
        pathology_files: List of pathology file paths
        ultrasound_files: List of ultrasound file paths
        pathology_key_col: Column name for pathology patient ID
        ultrasound_key_col: Column name for ultrasound patient ID
        filter_by_hasvideo: Whether to filter ultrasound data by g_columnn_name_hasVideo column (default: False)
        custom_filters: List of custom filters to apply (column_name, operator, value)
    
    Returns:
        Tuple of (final merged DataFrame, statistics dictionary)
    """
    logger = logging.getLogger(__name__)
    
    # Load and combine all pathology files
    combined_pathology_df, _ = combine_files(pathology_files, "pathology")
    
    # Load and combine all ultrasound files
    combined_ultrasound_df, ultrasound_dfs = combine_files(ultrasound_files, "ultrasound")
    original_ultrasound_count = len(combined_ultrasound_df)
    
    # Apply filters to ultrasound data
    filtered_ultrasound_df, filteredOut_by_condition = apply_data_filters(
        combined_ultrasound_df, custom_filters, filter_by_hasvideo
    )
    
    # Match the data
    final_df, filtered_count = match_dataframes(
        combined_pathology_df, 
        filtered_ultrasound_df, 
        pathology_key_col, 
        ultrasound_key_col
    )
    
    # Generate statistics
    stats = generate_statistics(
        combined_pathology_df, filtered_ultrasound_df, original_ultrasound_count, 
        ultrasound_dfs, pathology_key_col, ultrasound_key_col, 
        filtered_count, final_df, filter_by_hasvideo, filteredOut_by_condition
    )
    
    return final_df, stats


def parse_custom_filter(filter_string: str) -> Tuple[str, str, str]:
    """
    Parse a custom filter string in format "column:operator:value"
    Supported operators: eq (equals), neq (not equals), contains (substring), not_contains (exclusion)
    
    Args:
        filter_string: String in format "column:operator:value"
    
    Returns:
        Tuple of (column_name, operator, value)
    """
    parts = filter_string.split(':', 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid filter format: {filter_string}. Expected format: column:operator:value")
    
    column_name, operator, value = parts
    if operator not in ['eq', 'neq', 'contains', 'not_contains']:
        raise ValueError(f"Invalid operator: {operator}. Supported operators: eq, neq, contains, not_contains")
    
    return column_name, operator, value


def generate_output_filename(original_filename: str) -> str:
    """
    Generate output filename with timestamp suffix in format '_yymmddThhmmss'.
    
    Args:
        original_filename: The original output filename
        
    Returns:
        New filename with timestamp suffix
    """
    timestamp = datetime.now().strftime("%y%m%dT%H%M%S")
    output_path = Path(original_filename)
    return f"{output_path.stem}_{timestamp}{output_path.suffix}"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Match pathology and ultrasound spreadsheet files based on patient IDs"
    )
    
    parser.add_argument(
        '--pathology-files', 
        nargs='+', 
        required=True,
        help="List of pathology spreadsheet files"
    )
    
    parser.add_argument(
        '--ultrasound-files', 
        nargs='+', 
        required=True,
        help="List of ultrasound spreadsheet files"
    )
    
    parser.add_argument(
        '--pathology-key-col', 
        default='病人编号',
        help="Column name for pathology patient ID (default: 病人编号)"
    )
    
    parser.add_argument(
        '--ultrasound-key-col', 
        default='病人ID',
        help="Column name for ultrasound patient ID (default: 病人ID)"
    )
    
    parser.add_argument(
        '--filter-by-hasvideo', 
        action='store_true',
        default=False,
        help=f"Filter ultrasound data: keep only records where {g_columnn_name_hasVideo} column value is '有' (default: False)"
    )
    
    parser.add_argument(
        '--custom-filter',
        action='append',
        dest='custom_filters',
        metavar='COLUMN:OPERATOR:VALUE',
        help="Custom filters in format 'column:operator:value' where operator is 'eq', 'neq', 'contains', or 'not_contains'. "
             "Can be used multiple times. Examples: --custom-filter '检查项目:contains:引导' or --custom-filter '检查项目:not_contains:引导'"
    )
    
    parser.add_argument(
        '--output-file', 
        default='results_UsPatho.xlsx',
        help="Output file name (default: results_UsPatho.xlsx)"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    logger = setup_logging()
    logger.info("Starting spreadsheet matching application...")
    
    # Process custom filters
    custom_filters = []
    if args.custom_filters:
        for filter_str in args.custom_filters:
            try:
                column_name, operator, value = parse_custom_filter(filter_str)
                custom_filters.append((column_name, operator, value))
                logger.info(f"Added custom filter: {column_name} {operator} '{value}'")
            except ValueError as e:
                logger.error(f"Error parsing custom filter '{filter_str}': {e}")
                sys.exit(1)
    
    try:
        # Process the files
        final_df, stats = process_files(
            args.pathology_files,
            args.ultrasound_files,
            args.pathology_key_col,
            args.ultrasound_key_col,
            args.filter_by_hasvideo,
            custom_filters
        )
        
        # Generate output file name with timestamp
        output_file_with_timestamp = generate_output_filename(args.output_file)
        
        # Save the result
        logger.info(f"Saving matched results to {output_file_with_timestamp}")
        final_df.to_excel(output_file_with_timestamp, index=False)
        logger.info(f"Results saved successfully to {output_file_with_timestamp}")
        
        print_statistics(stats)
        
        logger.info("Application completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()