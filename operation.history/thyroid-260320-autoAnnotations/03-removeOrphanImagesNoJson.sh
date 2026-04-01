#!/usr/bin/env bash

set -euo pipefail

# ----------------------
# Configuration
# ----------------------

# Supported image extensions (case‑insensitive)
IMAGE_EXTENSIONS=("jpg" "jpeg" "png" "gif" "bmp" "tif" "tiff" "webp" "dcm")

# Log file prefix (will be timestamped in the current directory)
LOG_PREFIX="removed_images"

# ----------------------
# Helper functions
# ----------------------

# Convert a string to lower case
to_lower() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

# Check if a file exists (case‑sensitive)
file_exists() {
    [[ -f "$1" ]]
}

# Log a message to both stdout and the log file
log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] $*" | tee -a "$log_file"
}

# ----------------------
# Core functions
# ----------------------

# Scan the given directory and collect image files without a matching .json file.
# Returns a newline‑separated list of relative filenames (via stdout).
get_files_to_remove() {
    local dir="$1"
    local -a to_remove=()
    local file base ext ext_lower

    # Change to the directory to work with relative names
    cd "$dir" || return 1

    for file in *; do
        [ -f "$file" ] || continue

        ext="${file##*.}"
        ext_lower="$(to_lower "$ext")"

        # Check if the file is an image (extension in the allowed list)
        local is_image=false
        for allowed in "${IMAGE_EXTENSIONS[@]}"; do
            if [ "$ext_lower" = "$allowed" ]; then
                is_image=true
                break
            fi
        done
        [ "$is_image" = false ] && continue

        base="${file%.*}"
        if ! file_exists "${base}.json"; then
            to_remove+=("$file")
        fi
    done

    # Output each file on its own line
    printf '%s\n' "${to_remove[@]}"
}

# Remove a list of files (provided as arguments) and log each removal.
# Returns the number of files that were successfully removed.
remove_files() {
    local -a files=("$@")
    local removed_count=0

    for file in "${files[@]}"; do
        full_path="$input_dir/$file"
        if rm -f "$full_path"; then
            log "Removed: $full_path"
            ((removed_count++))
        else
            log "ERROR: Failed to remove $full_path"
        fi
    done

    echo "$removed_count"
}

# ----------------------
# Main function
# ----------------------

main() {
    # Argument handling
    if [ $# -ne 1 ]; then
        echo "Usage: $0 <input_directory>"
        echo "  Removes image files without a matching .json file (same base name)."
        exit 1
    fi

    local input_dir="$1"
    if [ ! -d "$input_dir" ]; then
        echo "Error: '$input_dir' is not a valid directory." >&2
        exit 1
    fi

    # Create a timestamped log file
    local log_file="${LOG_PREFIX}_$(date '+%Y%m%d_%H%M%S').log"
    touch "$log_file" || {
        echo "Error: Cannot create log file '$log_file'." >&2
        exit 1
    }

    # Collect the files to remove
    echo "Scanning '$input_dir' for images without a matching .json file..."
    local -a to_remove=()
    mapfile -t to_remove < <(get_files_to_remove "$input_dir")

    if [ ${#to_remove[@]} -eq 0 ]; then
        echo "No image files without a matching JSON file found in '$input_dir'."
        exit 0
    fi

    # Show the list and ask for confirmation
    echo "The following image files will be removed (they have no matching .json file):"
    printf "  %s\n" "${to_remove[@]}"
    echo

    read -r -p "Are you sure you want to remove these ${#to_remove[@]} file(s)? (y/N) " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted by user."
        exit 0
    fi

    # Perform the removal and log
    log "Starting removal of ${#to_remove[@]} files from '$input_dir'"
    local removed_count
    removed_count=$(remove_files "${to_remove[@]}")
    log "Removal completed. Successfully removed $removed_count files."

    echo "Done. Removed $removed_count files. See $log_file for details."
}

# ----------------------
# Execute main
# ----------------------

main "$@"
