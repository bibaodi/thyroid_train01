"""
CSV File Merger Utility
Author: Eton
First Edition: May 2024

This script merges two CSV files while ensuring unique entries in a specified primary key column.
Automatically generates output filename if not provided, and produces a merge report.
"""

import pandas as pd
import argparse
from pathlib import Path

def generate_output_filename(file1: str, file2: str) -> str:
    """Generate default output filename from input filenames"""
    file1_stem = Path(file1).stem
    file2_stem = Path(file2).stem
    return f"combined_{file1_stem}_{file2_stem}.csv"

def merge_csv_files(file1, file2, key_column, output_file):
    """Merge two CSV files while ensuring unique primary keys"""
    try:
        # Read CSV files
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Generate report statistics
        orig_count1 = len(df1)
        orig_count2 = len(df2)
        total_before = orig_count1 + orig_count2

        # Validate key column exists
        for df, name in [(df1, file1), (df2, file2)]:
            if key_column not in df.columns:
                raise ValueError(f"Column '{key_column}' not found in {name}")

        # Combine and deduplicate
        combined = pd.concat([df1, df2])
        combined.drop_duplicates(subset=[key_column], keep='first', inplace=True)

        # Generate final counts
        final_count = len(combined)
        duplicates_removed = total_before - final_count

        # Save result
        combined.to_csv(output_file, index=False)
        
        # Print merge report
        print(f"Merge Report:\n"
              f"- Original items in {Path(file1).name}: {orig_count1}\n"
              f"- Original items in {Path(file2).name}: {orig_count2}\n"
              f"- Total items before merge: {total_before}\n"
              f"- Final items after deduplication: {final_count}\n"
              f"- Duplicates removed: {duplicates_removed}\n"
              f"Output saved to: {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

def main():
    """Main entry point for command line execution"""
    parser = argparse.ArgumentParser(description='CSV File Merger')
    parser.add_argument('--file1', required=True, help='First input CSV file')
    parser.add_argument('--file2', required=True, help='Second input CSV file')
    parser.add_argument('--key', required=True, dest='key_column',
                      help='Column name to use as primary key')
    parser.add_argument('--output', help='Output CSV file path (default: generate from input names)')
    
    args = parser.parse_args()
    args.output = args.output or generate_output_filename(args.file1, args.file2)
    merge_csv_files(args.file1, args.file2, args.key_column, args.output)

if __name__ == "__main__":
    main()