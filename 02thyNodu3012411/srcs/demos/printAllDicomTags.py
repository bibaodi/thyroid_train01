import sys,os
import pydicom

def print_dicom_tagsV1(dicom_file_path):
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_file_path)

    # Iterate over all elements in the dataset
    for elem in ds:
        # Print group and element (tag) in hex, name, and value
        print(f"Group: 0x{elem.tag.group:04x}, Element: 0x{elem.tag.elem:04x}, "
              f"Name: {elem.name}, Value: {elem.value}")
def map_transfer_syntax(uid):
    # Mapping of common Transfer Syntax UIDs to human-readable names
    syntax_map = {
        "1.2.840.10008.1.2.1": "Little Endian Explicit",
        "1.2.840.10008.1.2.2": "Little Endian Implicit",
        "1.2.840.10008.1.2.4.50": "JPEG Baseline (Process 1)",
        "1.2.840.10008.1.2.4.51": "JPEG Extended (Process 2 & 4)",
        "1.2.840.10008.1.2.4.57": "JPEG Lossless, Non-Hierarchical (Process 14)",
        "1.2.840.10008.1.2.4.70": "JPEG Lossless, Non-Hierarchical (Process 14 [Selection Value 1])",
        "1.2.840.10008.1.2.4.80": "JPEG-LS Lossless",
        "1.2.840.10008.1.2.4.81": "JPEG-LS Lossy",
        "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
        "1.2.840.10008.1.2.4.91": "JPEG 2000",
        "1.2.840.10008.1.2.4.92": "JPEG 2000 Part 2 (Multi-component)",
        "1.2.840.10008.1.2.5": "RLE Lossless",
        "1.2.840.10008.1.2.4.100": "MPEG2 Main Profile @ Main Level",
        "1.2.840.10008.1.2.4.101": "MPEG2 Main Profile @ High Level",
        "1.2.840.10008.1.2.4.102": "MPEG-4 AVC/H.264 High Profile / Level 4.1",
        "1.2.840.10008.1.2.4.103": "MPEG-4 AVC/H.264 BD-compatible High Profile / Level 4.1",
        "1.2.840.10008.1.2.4.104": "MPEG-4 AVC/H.264 High Profile / Level 4.2 For 2D Video",
        "1.2.840.10008.1.2.4.105": "MPEG-4 AVC/H.264 High Profile / Level 4.2 For 3D Video",
        "1.2.840.10008.1.2.4.106": "MPEG-4 AVC/H.264 Stereo High Profile / Level 4.2",
        "1.2.840.10008.1.2.4.93": "JPEG 2000 Part 2 (Multi-component) Lossless",
        "1.2.840.10008.1.2.4.94": "JPEG 2000 Part 2 (Multi-component)",
        "1.2.840.10008.1.2.4.95": "JPIP Referenced",
        "1.2.840.10008.1.2.4.96": "JPIP Referenced Deflate",
        "1.2.840.10008.1.2.6.1": "Deflated Explicit VR Little Endian",
        "1.2.840.10008.1.2.6.2": "Deflated Explicit VR Big Endian",
    }
    return syntax_map.get(uid, uid)

def print_dicom_tags(dicom_file_path):
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_file_path)

    # Print Transfer Syntax UID and its mapped name
    transfer_syntax_uid = ds.file_meta.TransferSyntaxUID
    print(f"Transfer Syntax UID: {transfer_syntax_uid} ({map_transfer_syntax(transfer_syntax_uid)})")


    # Iterate over all elements in the dataset
    for elem in ds:
        group = f"0x{elem.tag.group:04x}"
        element = f"0x{elem.tag.elem:04x}"
        name = elem.name
        value = elem.value

        # If the value is binary or too long, print its length instead
        if elem.VR == "OB" or elem.VR == "OW" or elem.VR == "OF" or (isinstance(value, bytes) or len(str(value)) > 64):
            value_repr = f"<length: {len(value)}>"
        else:
            value_repr = str(value)

        print(f"Group: {group}, Element: {element}, Name: {name}, Value: {value_repr}")

if __name__ == "__test__":
    dicom_file_path = input("Enter the path to your DICOM file: ")
    print_dicom_tags(dicom_file_path)

def main():
    # Get file path from command line or prompt user
    if len(sys.argv) > 1:
        dicom_file_path = sys.argv[1]
    else:
        dicom_file_path = None

    while True:
        if dicom_file_path is None or not os.path.isfile(dicom_file_path):
            dicom_file_path = input("Enter the path to your DICOM file: ").strip()

        if os.path.isfile(dicom_file_path):
            print_dicom_tags(dicom_file_path)
            break
        else:
            print("File not found or not readable. Please try again.")
            dicom_file_path = None

if __name__ == "__main__":
    main()
