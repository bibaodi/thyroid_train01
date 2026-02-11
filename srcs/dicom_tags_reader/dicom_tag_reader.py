#!/usr/bin/env python3
"""
DICOM Tag Reader - Print all DICOM tag IDs, names, and values
"""

import os
import sys
from pathlib import Path

try:
    import pydicom
except ImportError:
    print("pydicom library is required but not installed.")
    print("Please install it using: pip install pydicom")
    sys.exit(1)


def print_dicom_tags(file_path):
    """
    Read a DICOM file and print all tags with their IDs, names, and values
    """
    try:
        # Load the DICOM file
        ds = pydicom.dcmread(file_path)
        
        print(f"\nDICOM File: {file_path}")
        print("=" * 120)
        print(f"{'Tag':<20} {'Name':<30} {'Type':<25} {'Value'}")
        print("-" * 120)
        
        # Iterate through all data elements in the dataset
        for elem in ds:
            # Get tag in (group, element) format
            tag = f"({elem.tag.group:04X},{elem.tag.element:04X})"
            
            # Get element name (if available)
            try:
                name = elem.name
            except:
                name = str(elem.keyword) if elem.keyword else "Unknown"
            
            # Get DICOM VR (Value Representation) and Python type
            dicom_vr = str(elem.VR) if hasattr(elem, 'VR') else "Unknown"
            python_type = type(elem.value).__name__
            combined_type = f"{dicom_vr}:{python_type}"
            
            # Handle binary data differently
            if isinstance(elem.value, bytes):
                value = f"<binary data length: {len(elem.value)}>"
            elif hasattr(elem.value, '__len__') and hasattr(elem.value, '__iter__') and not isinstance(elem.value, (str, bytes)):
                # Special handling for PersonName since it's a common case
                if python_type == 'PersonName':
                    # PersonName should be treated as a string-like object
                    value = str(elem.value)
                    # Replace newlines with spaces to prevent formatting issues
                    value = value.replace('\n', ' ').replace('\r', ' ')
                    if len(value) > 100:
                        value = f"[length: {len(value)}, content: {value[:50].replace(chr(10), ' ').replace(chr(13), ' ')}...[truncated]]"
                else:
                    # Handle sequence/array data - show length and first 3 items if applicable
                    try:
                        length = len(elem.value)
                        if length > 3:
                            # Show first 3 items for long sequences
                            first_three = ', '.join(str(item)[:30].replace('\n', ' ').replace('\r', ' ') for item in elem.value[:3])
                            value = f"[length: {length}, first 3: {first_three}...]"
                        else:
                            # For short sequences, show all items
                            all_items = ', '.join(str(item)[:30].replace('\n', ' ').replace('\r', ' ') for item in elem.value)
                            value = f"[length: {length}, items: {all_items}]"
                    except:
                        # Fallback for complex sequences
                        value = f"<sequence with {len(elem.value)} items>"
            else:
                # For non-sequence values, convert to string and handle long values
                value = str(elem.value)
                # Replace newlines with spaces to prevent formatting issues
                value = value.replace('\n', ' ').replace('\r', ' ')
                if len(value) > 100:
                    value = f"[length: {len(value)}, content: {value[:50].replace(chr(10), ' ').replace(chr(13), ' ')}...[truncated]]"
            
            print(f"{tag:<20} {name:<30} {combined_type:<25} {value}")
        
        print("\n" + "=" * 120)
        
    except Exception as e:
        print(f"Error reading DICOM file {file_path}: {str(e)}")


def print_vr_explanations(dicom_files, vr_meanings):
    """
    Print explanations about DICOM Value Representations (VRs) and list all VRs found in the files.
    
    DICOM VRs (Value Representations) define the data type and format of DICOM attribute values.
    Each DICOM tag has a specific VR that tells the system how to interpret the data.
    For example:
    - CS (Code String): Used for standardized coded values
    - DA (Date): Represents dates in YYYYMMDD format
    - LO (Long String): Used for longer text descriptions
    - UI (Unique Identifier): Globally unique identifiers like UIDs
    """
    print("\n" + "="*80)
    print("DICOM VALUE REPRESENTATION (VR) EXPLANATION")
    print("="*80)
    print("DICOM VRs define how data should be interpreted in DICOM files.")
    print("Each DICOM tag has a specific VR indicating the data type and format.\n")
    
    print("COMMON DICOM VR EXAMPLES:")
    # Use the VR meanings dictionary to get the explanations dynamically
    common_vrs = ['CS', 'DA', 'DS', 'IS', 'LO', 'PN', 'UI', 'US', 'UL', 'OB', 'SQ']
    for vr in common_vrs:
        meaning = vr_meanings.get(vr, "Unknown VR")
        # Add example descriptions for common VRs
        examples = {
            'CS': "(e.g., 'M' for male)",
            'DA': "(in YYYYMMDD format)",
            'DS': "(e.g., '1.5', '3.14159')",
            'IS': "(e.g., '128', '1024')",
            'LO': "(up to 64 chars of text)",
            'PN': "(patient names in special format)",
            'UI': "(globally unique identifiers)",
            'US': "(16-bit unsigned integers)",
            'UL': "(32-bit unsigned integers)",
            'OB': "(raw binary data, often pixel data)",
            'SQ': "(nested sequences of DICOM objects)"
        }
        example = examples.get(vr, "")
        print(f"  {vr} ({meaning}): {example}".ljust(50))
    print()
    
    print("HOW TO USE THIS OUTPUT:")
    print("  - The main table shows: Tag ID | Name | VR:PythonType | Value")
    print("  - VR:PythonType shows both DICOM VR and the Python data type")
    print("  - Understanding VRs helps interpret what kind of data each tag contains")
    print("  - Different VRs have different storage and interpretation rules\n")
    
    # Collect all unique VRs found in the files
    all_found_vrs = set()
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_file)
            for elem in ds:
                if hasattr(elem, 'VR'):
                    all_found_vrs.add(str(elem.VR))
        except:
            continue
    
    # Print VR meanings for those found in the file
    if all_found_vrs:
        print("DICOM VR MEANINGS FOUND IN YOUR FILE(S):")
        print("-" * 60)
        for vr in sorted(all_found_vrs):
            meaning = vr_meanings.get(vr, "Unknown VR")
            print(f"  {vr}: {meaning}")
    else:
        print("No DICOM VRs found in the processed files.")
    
    print("\nFor more information about DICOM standards, visit: https://www.dicomstandard.org/")
    print("="*80)


def get_vr_meanings():
    """
    Returns a dictionary mapping DICOM Value Representations (VRs) to their meanings.
    These are standardized abbreviations that indicate how DICOM data should be interpreted.
    """
    return {
        'AE': 'Application Entity',
        'AS': 'Age String',
        'AT': 'Attribute Tag',
        'CS': 'Code String',
        'DA': 'Date',
        'DS': 'Decimal String',
        'DT': 'Date Time',
        'FL': 'Floating Point Single',
        'FD': 'Floating Point Double',
        'IS': 'Integer String',
        'LO': 'Long String',
        'LT': 'Long Text',
        'OB': 'Other Byte',
        'OD': 'Other Double',
        'OF': 'Other Float',
        'OW': 'Other Word',
        'PN': 'Person Name',
        'SH': 'Short String',
        'SL': 'Signed Long',
        'SQ': 'Sequence of Items',
        'SS': 'Signed Short',
        'ST': 'Short Text',
        'TM': 'Time',
        'UC': 'Unlimited Characters',
        'UI': 'Unique Identifier',
        'UL': 'Unsigned Long',
        'UN': 'Unknown',
        'UR': 'Universal Resource',
        'US': 'Unsigned Short',
        'UT': 'Unlimited Text'
    }


def main():
    vr_meanings = get_vr_meanings()
    
    # Get the current directory
    current_dir = Path(".")
    
    # Find all DICOM files in the current directory
    dicom_extensions = [".dcm", ".DCM", ".dicom", ".DICOM"]
    dicom_files = []
    
    for ext in dicom_extensions:
        dicom_files.extend(list(current_dir.glob(f"*{ext}")))
    
    # Also look for DICOM files without extensions that might be DICOM
    for item in current_dir.iterdir():
        if item.is_file() and not item.suffix:
            # Try to identify if it's a DICOM file by content
            try:
                with open(item, 'rb') as f:
                    header = f.read(132)  # DICOM header is 132 bytes
                    if len(header) >= 132 and header[128:132] == b'DICM':
                        dicom_files.append(item)
            except:
                pass
    
    if not dicom_files:
        print("No DICOM files found in the current directory.")
        print("Looking for files with extensions:", ", ".join(dicom_extensions))
        return
    
    print(f"Found {len(dicom_files)} DICOM file(s):\n")
    
    # Process each DICOM file
    for dicom_file in dicom_files:
        print_dicom_tags(dicom_file)
    
    # Print DICOM VR explanations and found VRs
    print_vr_explanations(dicom_files, vr_meanings)


if __name__ == "__main__":
    main()