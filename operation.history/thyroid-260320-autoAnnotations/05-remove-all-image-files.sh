#!/bin/bash

# -----------------------------------------------------------------------------
# Script: remove_images.sh
# Description: Recursively find and remove common image files.
# Usage: ./remove_images.sh [--dry-run] [directory]
# -----------------------------------------------------------------------------

# Default directory
TARGET_DIR="."

# Parse arguments
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            TARGET_DIR="$1"
            shift
            ;;
    esac
done

# Define image extensions (case-insensitive)
EXTENSIONS=(
    "jpg" "jpeg" "png" "gif" "bmp"
    "tiff" "tif" "webp" "svg" "ico"
    "heic" "heif" "raw" "cr2" "nef"
)

# Build the find arguments as an array (safer than eval)
find_args=("$TARGET_DIR" -type f)
for ext in "${EXTENSIONS[@]}"; do
    find_args+=(-iname "*.$ext" -o)
done
# Remove the trailing "-o" (the last element)
unset 'find_args[-1]'

# If dry-run, just print the files
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry-run mode: no files will be deleted."
    find "${find_args[@]}" -print
    exit 0
fi

# Show the files that will be deleted
echo "The following files will be deleted:"
find "${find_args[@]}" -print
echo
read -p "Are you sure you want to delete these files? (y/N): " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Deleting files..."
    # Use -exec rm {} + (works on all find versions)
    find "${find_args[@]}" -exec rm {} +
    echo "Done."
else
    echo "Aborted."
fi
