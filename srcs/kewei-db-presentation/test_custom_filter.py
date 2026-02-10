#!/usr/bin/env python3
"""
Test script to demonstrate the new custom filter functionality
"""
import pandas as pd
from match_spreadsheets import apply_filters

def test_custom_filters():
    # Create a sample DataFrame similar to what might be in the spreadsheets
    data = {
        '检查项目': ['引导A超声', '普通B超', '引导彩色多普勒', '常规检查', '引导病理'],
        '病人编号': ['P001', 'P002', 'P003', 'P004', 'P005'],
        '录像': ['有', '无', '有', '有', '无'],
        '结果': ['正常', '异常', '正常', '待定', '异常']
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Test 1: Filter OUT rows where '检查项目' contains '引导' (using not_contains operator)
    print("Test 1: Filtering OUT rows where '检查项目' contains '引导' (using not_contains operator):")
    filters = [('检查项目', 'not_contains', '引导')]
    filtered_df = apply_filters(df.copy(), filters)
    print("After filtering (keeping records that do NOT contain '引导'):")
    print(filtered_df)
    print()
    
    # Test 2: Filter to keep only rows where '检查项目' contains '引导' (using contains operator)
    print("Test 2: Keeping only rows where '检查项目' contains '引导' (using contains operator):")
    filters = [('检查项目', 'contains', '引导')]
    filtered_df2 = apply_filters(df.copy(), filters)
    print("After filtering (keeping records that DO contain '引导'):")
    print(filtered_df2)
    print()
    
    # Test 3: Multiple filters
    print("Test 3: Applying multiple filters:")
    print("Filter 1: Keep records that do NOT contain '引导' in '检查项目'")
    print("Filter 2: Keep records where '录像' equals '有'")
    multi_filters = [
        ('检查项目', 'not_contains', '引导'),
        ('录像', 'eq', '有')
    ]
    multi_filtered_df = apply_filters(df.copy(), multi_filters)
    print("After multiple filters:")
    print(multi_filtered_df)
    print()

if __name__ == "__main__":
    test_custom_filters()