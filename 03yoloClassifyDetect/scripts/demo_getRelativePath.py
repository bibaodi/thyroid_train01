import os
from pathlib import Path

def get_relative_path(absolute_path, base_dir):
    # Convert the base directory to an absolute path
    base_dir_abs = Path(base_dir).resolve()

    # Compute the relative path
    relative_path = os.path.relpath(absolute_path, start=base_dir_abs)

    return relative_path

def get_relative_path_to_labels(absolute_path, base_dir_name):
    # Find the position of the base directory in the absolute path
    parts = Path(absolute_path).parts
    try:
        base_index = parts.index(base_dir_name)
    except ValueError:
        raise ValueError(f"The base directory '{base_dir_name}' is not found in the path.")

    # Construct the base directory path
    base_dir_path = Path(*parts[:base_index + 1])

    # Compute the relative path from the base directory
    relative_path = os.path.relpath(absolute_path, start=base_dir_path)

    return relative_path
# Example usage
absolute_path = "/a/b/c/labels/12/3/dajk.xx"
base_dir = "labels"

relative_path = get_relative_path(absolute_path, base_dir)
print("Relative Path:", relative_path)
relative_path = get_relative_path_to_labels(absolute_path, base_dir)
print("Relative Path:", relative_path)
