#!/bin/bash
#eton@250215 merge two labels(0,1)(Benign, Malign) to one(0=ThyNodu)
#eton@250315 add '-L' to find, otherwise the type f not work for symbolic link;
# Function to update YOLO label files
function update_yolo_labels() {
	local LOG_SKIP_FILE=0
	echo "LOG_SHIP_FILE=${LOG_SKIP_FILE};"
    local LABEL_DIR=$1
    # Check if the directory exists
    if [[ ! -d "$LABEL_DIR" ]]; then
        echo "The directory '$LABEL_DIR' does not exist."
        exit 1
    fi

# Confirm the directory with the user
read -p "You entered '$LABEL_DIR'. Is this correct? (y/n): " confirm

if [[ $confirm != [yY] ]]; then
    echo "Operation cancelled."
    exit 0
fi

    # Find all .txt files in the directory and its subdirectories
    find -L "$LABEL_DIR" -type f -name "*.txt" | while read -r file; do
        # Check if the file contains the label '1'
        if grep -q '\b1\b' "$file"; then
            # Use sed to replace all occurrences of '1' with '0' in the file
            sed -i 's/\b1\b/0/g' "$file"
            echo "Processed $file"
        else
		test ${LOG_SKIP_FILE} -gt 0 && echo "Skipped $(basename $file) (no label '1' found)"
        fi

	# Use sed to replace all occurrences of '1' with '0' in the file
    done

    echo "All label files have been updated."
}

# Check if the first parameter is provided
if [[ -z "$1" ]]; then
    echo "Usage: $0 <label_dir>"
    exit 1
fi

# Call the function with the first parameter
update_yolo_labels "$1"

echo "All label files have been updated."
