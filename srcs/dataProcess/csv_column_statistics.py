#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSV Column Statistics Application
This script accepts a CSV file and column names as arguments,
then displays value counts for the specified columns using pandas.
"""

import argparse
import pandas as pd
import sys
import os
from io import StringIO

def read_csv_with_encoding_handling(file_path):
    """
    Read CSV file with encoding handling to avoid common encoding issues
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read CSV with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Failed to read CSV with encoding {encoding}: {str(e)}")
            continue
    
    raise ValueError(f"Could not read the CSV file with any of the attempted encodings: {encodings}")

def display_column_statistics_beautiful(df, column_names):
    """
    Display value counts for specified columns in the DataFrame in a beautiful format
    """
    for col in column_names:
        if col in df.columns:
            print(f"\n{col}")
            value_counts = df[col].value_counts()
            for index, count in value_counts.items():
                print(f"{index:>15} {count}")
        else:
            print(f"\nColumn '{col}' not found in the CSV file.")
            print(f"Available columns: {list(df.columns)}")

def display_column_statistics_csv(df, column_names, save_to_files=True):
    """
    Display value counts for specified columns in the DataFrame in CSV format
    Optionally save to files with auto-generated names
    """
    csv_outputs = []
    
    for col in column_names:
        if col in df.columns:
            output = StringIO()
            value_counts = df[col].value_counts()
            # Write the column name as a header
            output.write("type,count\n")
            for index, count in value_counts.items():
                # Properly quote the type value to handle commas, quotes, etc.
                safe_index = str(index).replace('"', '""')  # Escape quotes
                output.write(f'"{safe_index}",{count}\n')
            
            csv_content = output.getvalue()
            output.close()
            
            csv_outputs.append({
                'column': col,
                'content': csv_content
            })
            
            print(f"\n--- {col} ---")
            print(csv_content.strip())
        else:
            error_content = f"type,count\n\"Error: Column '{col}' not found\",0\n"
            csv_outputs.append({
                'column': col,
                'content': error_content
            })
            print(f"\n--- {col} ---")
            print(error_content.strip())
    
    # Save to auto-generated files
    if save_to_files:
        for csv_output in csv_outputs:
            output_filename = f"label_stats_{csv_output['column']}.csv"
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(csv_output['content'])
            print(f"\nCSV output for '{csv_output['column']}' saved to: {output_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="""Display value counts for specified columns in a CSV file in both beautiful and CSV formats;
        csv_column_statistics.py tirads_matched_sops_v2-eton-purged251226.csv composition echo shape margin
        """
    )
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('columns', nargs='+', help='Column names to analyze')
    parser.add_argument('--format', choices=['beautiful', 'csv', 'both'], default='both',
                        help='Output format: beautiful (formatted table), csv (CSV format), or both (default)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Read CSV file
        df = read_csv_with_encoding_handling(args.csv_file)
        
        # Display statistics based on format argument
        if args.format in ['beautiful', 'both']:
            print("Beautiful Format:")
            display_column_statistics_beautiful(df, args.columns)
        
        if args.format in ['csv', 'both']:
            if args.format == 'both':
                print("\n" + "="*50)
                print("CSV Format:")
            
            display_column_statistics_csv(df, args.columns, save_to_files=True)
        
    except Exception as e:
        print(f"Error processing the CSV file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()