#!/bin/bash

# Directory path
root_directory="/path/to/your/directory"
root_directory=${1:-'nowhere'}

# Function to count files in a directory and its subdirectories
count_files() {
    local dir=$1
    local total_files=0

    # Count files in the current directory
    local current_files=$(find -L "$dir"   -maxdepth 1 -type f | wc -l)
    total_files=$((total_files + current_files))

    # Recursively count files in subdirectories
    while IFS= read -r -d '' subdir; do
        local subdir_files=$(count_files "$subdir")
        total_files=$((total_files + subdir_files))
    done < <(find "$dir" -mindepth 1 -maxdepth 1 -type d -print0)

    echo "$total_files"
}

# Traverse the root directory and count files
find "$root_directory" -type d -print0 | while IFS= read -r -d '' dir; do
    count=$(count_files "$dir")
    echo "Directory: $dir, Total File count: $count"
done



