#!/usr/bin/env bash

# Exit on undefined variables, but allow commands to fail so we can handle errors gracefully.
set -u

# ----------------------
# Configuration
# ----------------------

# List of image file extensions to copy (case‑sensitive – adjust as needed)
IMAGE_EXTENSIONS=("*.jpg" "*.jpeg" "*.png" "*.tif" "*.tiff" "*.bmp" "*.dcm" "*.DCM")

# ----------------------
# Helper functions
# ----------------------

# Convert a path to an absolute path.
to_absolute_path() {
    local path="$1"
    if command -v realpath &>/dev/null; then
        realpath "$path"
    elif command -v readlink &>/dev/null; then
        readlink -f "$path"
    else
        # Fallback: use cd and pwd (requires path to exist or we cannot resolve)
        if [ -e "$path" ]; then
            (cd "$(dirname "$path")" && echo "$(pwd)/$(basename "$path")")
        else
            echo "$path"
        fi
    fi
}

# Log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Show help message
show_help() {
    cat <<EOF
Usage: $0 [options] <input_home> <output_home>

Processes all first‑level subfolders of <input_home> with an ML model.

Options:
  -c, --copy-mode <after|only>   Copy images:
                                 after   – run annotation, then copy images
                                 only    – copy images only (skip annotation)
                                 default – do not copy images;
  -h, --help                     Show this help message

The script will ask for confirmation (y/n/a/q) for each subfolder.
EOF
}

# Run the annotation command for one subfolder.
run_annotation() {
    local input_folder="$1"
    local output_folder="$2"

    local cmd=(
        conda run -n trainLinkNet18_env python /home/eton/00-srcs/ml_thyroid.01_etonTrainCodes/srcs/autoAnnotate/annotateByMLModel.py
        --model_type segmentation
        --model_file /mnt/datas/42workspace/34-project_ML_data_models_UltrasoundIntelligence/44-models/3-thyroid/model_segmentThyGland_v02.250821/model_segmentThyGland_v02.250821.pt
        --label_name ThyGland
        --input_folder "$input_folder"
        --output_folder "$output_folder"
    )

    log "Running: ${cmd[*]}"
    "${cmd[@]}"
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log "Successfully processed '$input_folder'."
    else
        log "ERROR: Processing '$input_folder' failed with exit code $exit_code."
    fi
    return $exit_code
}

# Copy all image files from source folder to destination folder.
copy_images() {
    local src="$1"
    local dst="$2"

    # Enable nullglob to avoid literal globs when no files match
    shopt -s nullglob
    local copied=0
    for ext in "${IMAGE_EXTENSIONS[@]}"; do
        for file in "$src"/$ext; do
            # Copy with -n (no clobber) to avoid overwriting
            cp -n "$file" "$dst/"
            if [ $? -eq 0 ]; then
                ((copied++))
            fi
        done
    done
    shopt -u nullglob

    log "Copied $copied image(s) from '$src' to '$dst'."
}

# Generate the confirmation prompt message based on copy_mode.
get_prompt_message() {
    local subfolder="$1"
    case "${copy_mode:-}" in
        after)
            echo "Process subfolder '$subfolder'? (annotation then copy images) [y/n/a/q] "
            ;;
        only)
            echo "Copy images only for subfolder '$subfolder'? [y/n/a/q] "
            ;;
        *)
	    echo "(no copy)Process subfolder '$subfolder'? [y/n/a/q] "
            ;;
    esac
}

# ----------------------
# Argument parsing
# ----------------------

copy_mode=""   # can be "after", "only", or empty (no copy)

# Use getopts for short options; long options can be emulated with a separate loop if needed.
# Here we accept -c and -h.
while getopts "c:h" opt; do
    case "$opt" in
        c)
            copy_mode="$OPTARG"
            # Validate mode
            if [[ ! "$copy_mode" =~ ^(after|only)$ ]]; then
                echo "Error: --copy-mode must be 'after' or 'only'." >&2
                show_help
                exit 1
            fi
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

# Remaining arguments: input_home and output_home
if [ $# -ne 2 ]; then
    echo "Error: Missing required arguments." >&2
    show_help
    exit 1
fi

input_home="$(to_absolute_path "$1")"
output_home="$(to_absolute_path "$2")"

log "Input home resolved to: $input_home"
log "Output home resolved to: $output_home"
log "Copy mode: ${copy_mode:-none}"

# Validate input home directory
if [ ! -d "$input_home" ]; then
    echo "Error: Input home directory '$input_home' does not exist or is not a directory." >&2
    exit 1
fi

# Create output home directory if it doesn't exist
mkdir -p "$output_home" || {
    echo "Error: Cannot create output home directory '$output_home'." >&2
    exit 1
}

# ----------------------
# Main processing
# ----------------------

all_mode=0   # 0 = ask for each subfolder, 1 = process all without asking

# Use nullglob to avoid problems when there are no subdirectories
shopt -s nullglob

# Count total subfolders first
subfolders=("$input_home"/*/)
total_subfolders=${#subfolders[@]}
current_folder=0

log "Found $total_subfolders subfolder(s) to process."

for subfolder_path in "${subfolders[@]}"; do
    ((current_folder++))
    subfolder="$(basename "$subfolder_path")"
    input_subfolder="$input_home/$subfolder"
    output_subfolder="$output_home/$subfolder"

    log "[$current_folder/$total_subfolders] Found subfolder: $subfolder"

    # Create output subfolder (needed for both annotation and image copy)
    mkdir -p "$output_subfolder" || {
        log "ERROR: Cannot create output subfolder '$output_subfolder'. Skipping."
        continue
    }

    # Ask for confirmation unless already in 'all' mode
    if [ $all_mode -eq 0 ]; then
        prompt=$(get_prompt_message "$subfolder")
        while true; do
            read -r -p "$prompt" answer
            case "${answer,,}" in
                y|yes)
                    # Process according to copy_mode
                    case "${copy_mode:-}" in
                        after)
                            run_annotation "$input_subfolder" "$output_subfolder"
                            if [ $? -ne 0 ]; then
                                log "Exiting due to failure."
                                exit 1
                            fi
                            copy_images "$input_subfolder" "$output_subfolder"
                            ;;
                        only)
                            copy_images "$input_subfolder" "$output_subfolder"
                            ;;
                        *)
                            run_annotation "$input_subfolder" "$output_subfolder"
                            if [ $? -ne 0 ]; then
                                log "Exiting due to failure."
                                exit 1
                            fi
                            ;;
                    esac
                    break
                    ;;
                n|no)
                    log "Skipping subfolder '$subfolder'."
                    break
                    ;;
                a|all)
                    all_mode=1
                    # Process this subfolder now (since we're in 'all')
                    case "${copy_mode:-}" in
                        after)
                            run_annotation "$input_subfolder" "$output_subfolder"
                            if [ $? -ne 0 ]; then
                                log "Exiting due to failure."
                                exit 1
                            fi
                            copy_images "$input_subfolder" "$output_subfolder"
                            ;;
                        only)
                            copy_images "$input_subfolder" "$output_subfolder"
                            ;;
                        *)
                            run_annotation "$input_subfolder" "$output_subfolder"
                            if [ $? -ne 0 ]; then
                                log "Exiting due to failure."
                                exit 1
                            fi
                            ;;
                    esac
                    break
                    ;;
                q|quit)
                    log "Quitting by user request."
                    exit 0
                    ;;
                *)
                    echo "Please answer y, n, a, or q."
                    ;;
            esac
        done
    else
        # In 'all' mode: process without prompting
        case "${copy_mode:-}" in
            after)
                run_annotation "$input_subfolder" "$output_subfolder"
                if [ $? -ne 0 ]; then
                    log "Exiting due to failure."
                    exit 1
                fi
                copy_images "$input_subfolder" "$output_subfolder"
                ;;
            only)
                copy_images "$input_subfolder" "$output_subfolder"
                ;;
            *)
                run_annotation "$input_subfolder" "$output_subfolder"
                if [ $? -ne 0 ]; then
                    log "Exiting due to failure."
                    exit 1
                fi
                ;;
        esac
    fi
done

log "All done."