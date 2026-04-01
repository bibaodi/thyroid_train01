#!/usr/bin/env bash

# Exit on undefined variables, but allow commands to fail so we can handle errors gracefully.
set -u

# ----------------------
# Helper functions
# ----------------------

# Logs a message with a timestamp.
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Runs the annotation command for a given input and output folder.
# Returns the exit code of the Python script.
run_annotation() {
    local input_folder="$1"
    local output_folder="$2"

    # The full command to execute
    local cmd=(
        python /home/eton/00-srcs/ml_thyroid.01_etonTrainCodes/srcs/autoAnnotate/annotateByMLModel.py
        --model_type segmentation
        --model_file /mnt/datas/42workspace/34-project_ML_data_models_UltrasoundIntelligence/44-models/3-thyroid/model_segmentThyGland_v02.250821/model_segmentThyGland_v02.250821.pt
        --label_name thyGland
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

# ----------------------
# Argument handling
# ----------------------

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_home> <output_home>"
    echo "  input_home   - Directory containing subfolders with images and JSON files."
    echo "  output_home  - Directory where output subfolders will be created."
    exit 1
fi

input_home="$1"
output_home="$2"

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

all_mode=0  # 0 = ask for each subfolder, 1 = process all without asking

# Use nullglob so that the pattern */ expands to nothing if there are no subdirectories.
shopt -s nullglob

for subfolder_path in "$input_home"/*/; do
    # Extract the subfolder name (basename) without trailing slash
    subfolder="$(basename "$subfolder_path")"
    input_subfolder="$input_home/$subfolder"
    output_subfolder="$output_home/$subfolder"

    log "Found subfolder: $subfolder"

    # Create the corresponding output subfolder
    mkdir -p "$output_subfolder" || {
        log "ERROR: Cannot create output subfolder '$output_subfolder'. Skipping."
        continue
    }

    # Ask for confirmation unless in all mode
    if [ $all_mode -eq 0 ]; then
        while true; do
            read -r -p "Process subfolder '$subfolder'? [y/n/a/q] " answer
            case "${answer,,}" in
                y|yes)
                    # Process this subfolder
                    run_annotation "$input_subfolder" "$output_subfolder"
                    break
                    ;;
                n|no)
                    log "Skipping subfolder '$subfolder'."
                    break
                    ;;
                a|all)
                    all_mode=1
                    run_annotation "$input_subfolder" "$output_subfolder"
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
        # In all mode, process without prompting
        run_annotation "$input_subfolder" "$output_subfolder"
    fi
done

log "All done."
