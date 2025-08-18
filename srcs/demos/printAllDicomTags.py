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

def print_dicom_tags(dicom_file_path):
    # Load the DICOM file
    ds = pydicom.dcmread(dicom_file_path)

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
