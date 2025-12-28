import pandas as pd
import pathlib

class PositionFilter:
    def __init__(self, spreadsheet_path: str):
        self._validate_file(spreadsheet_path)
        self.df = pd.read_excel(
            spreadsheet_path,
            sheet_name='sop_0422',
            usecols=['sop_uid', 'position']
        )
        self.df['sop_uid'] = self.df['sop_uid'].str.lower()
        
    def _validate_file(self, path: str):
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"Position spreadsheet not found: {path}")
            
    def filter_by_position_prefix(self, prefix: str) -> pd.DataFrame:
        """Filter dataframe where position column starts with given prefix (case-insensitive)"""
        prefix = prefix.lower()
        return self.df[self.df['position'].str.lower().str.startswith(prefix)].copy()

# Example usage:
# filter = PositionFilter('path/to/excel.xlsx')
# xiabu_df = filter.filter_by_position_prefix('xiabu')
