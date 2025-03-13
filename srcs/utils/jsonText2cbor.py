import sys
import json
import cbor2
from pathlib import Path

def read_json_file(file_path):
    """
    Reads a JSON file and returns the data.

    :param file_path: Path to the JSON file.
    :return: Parsed JSON data.
    """
    try:
        with file_path.open('r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

def write_cbor_file(file_path, data):
    """
    Writes CBOR data to a file.

    :param file_path: Path to the output CBOR file.
    :param data: Data to be written as CBOR.
    """
    try:
        with file_path.open('wb') as file:
            file.write(data)
        print(f"CBOR data written to {file_path}")
    except Exception as e:
        print(f"Error writing CBOR file: {e}")
        sys.exit(1)

def convert_json_to_cbor(input_file, output_file):
    """
    Converts a JSON file to CBOR format.

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output CBOR file.
    """
    # Read JSON data
    json_data = read_json_file(input_file)

    # Encode JSON data to CBOR
    cbor_data = cbor2.dumps(json_data)

    # Write CBOR data to file
    write_cbor_file(output_file, cbor_data)

def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.json>")
        sys.exit(1)

    # Get the input file path from the command line
    input_file = Path(sys.argv[1])

    # Check if the input file exists
    if not input_file.is_file():
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)

    # Determine the output file path
    output_file = input_file.with_suffix('.cbor')

    # Convert JSON to CBOR
    convert_json_to_cbor(input_file, output_file)

if __name__ == "__main__":
    main()
