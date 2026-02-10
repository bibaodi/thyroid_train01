# Spreadsheet Matching Application

This Python application matches pathology and ultrasound spreadsheet files based on patient IDs.
```sh
python ./match_spreadsheets.py --ultrasound-files "2025.1.1-1.31甲状腺超声.xls" "2025.2.1-2.28甲状腺超声.xls" "2025.3.1-3.31甲状腺超声.xls" --pathology-files "2025.2.9-3.31病理结果表.xls" --filter-by-hasvideo --custom-filter '检查项目:not_contains:引导'
```

## Features

- Accepts multiple pathology and ultrasound spreadsheet files
- Matches records based on configurable patient ID column names
- Handles various Excel formats (.xls, .xlsx, .ods)
- Automatically detects and handles header rows in medical data files
- Generates a merged output file with matched records
- Creates detailed logs and statistics

## Requirements

- Python 3.7+
- pandas
- openpyxl
- odfpy
- xlrd

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python match_spreadsheets.py \
  --pathology-files <pathology_file1> [<pathology_file2> ...] \
  --ultrasound-files <ultrasound_file1> [<ultrasound_file2> ...] \
  --pathology-key-col "病人编号" \
  --ultrasound-key-col "病人ID" \
  --output-file "matched_results.xlsx"
```

### Arguments

- `--pathology-files`: List of pathology spreadsheet files (required)
- `--ultrasound-files`: List of ultrasound spreadsheet files (required)
- `--pathology-key-col`: Column name for pathology patient ID (default: "病人编号")
- `--ultrasound-key-col`: Column name for ultrasound patient ID (default: "病人ID")
- `--filter-by-hasvideo`: Filter ultrasound data to keep only records where '录像' column value is '有' (default: False, optional)
- `--output-file`: Output file name (default: "matched_results.xlsx")

### Example

```bash
python match_spreadsheets.py \
  --pathology-files "2025.2.9-3.31病理结果表.xls" \
  --ultrasound-files "2025.2.1-2.28甲状腺超声.xls" "2025.3.1-3.31甲状腺超声.xls" \
  --output-file "results.xlsx"
```

## Output

- Merged spreadsheet file with matched records
- Log file (`match_spreadsheets.log`) with detailed information
- Statistics showing:
  - Total items in pathology data
  - Total items in ultrasound data
  - Number of unique patients filtered out (no matching records)

## Notes

- The application handles special formatting in medical data files where the first row may contain metadata rather than headers
- When multiple pathology records exist for the same patient ID, the merge creates multiple rows per ultrasound record
- Output file contains all ultrasound records, with matching pathology data appended where available
