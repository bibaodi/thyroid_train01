#!/bin/bash
# author: eton.bi
# date: 2023-12-08
# description: copy all file from 301PACS-S02* folders to one home
function copySeperatesFile2OneHome_for301pxCasesHasTooMuchFolder(){
    # Source directory containing the 301PACS-S02* folders
    src_dir="/path/to/source"
    src_dir=$1

    # Destination directory (will be created if not exists)
    dest_dir="/path/to/destination/combined_files"
    dest_dir=$2

    MSG="will you copy all file from [${src_dir}] to [${dest_dir}] ?y/n"

    read -p "${MSG}" ans
    test ${ans} != 'y' && exit 0

    # Create destination directory
    mkdir -p "$dest_dir"

    # Find and process all folders starting with 301PACS-S02
    find "$src_dir" -type d -name "301PACS02*" -print0 | while IFS= read -r -d $'\0' folder; do
        # Get just the folder name
        folder_name=$(basename "$folder")
        # Process all files in this folder
        find "$folder" -type f -print0 | while IFS= read -r -d $'\0' file; do
            # Get just the filename
            filename=$(basename "$file")

            # Create new filename with folder prefix
            new_filename="${folder_name}_${filename}"

            # Copy file with new name
            cp -v "$file" "$dest_dir/$new_filename"

            # If file is JSON, update imagePath reference
            if [[ "$filename" == *.json ]]; then
                # Get the image extension from original JSON (png/jpg)
                img_ext=$(jq -r '.imagePath | split(".")[-1]' "$file")
                new_imagepath="${new_filename%.*}.$img_ext"
                
                # Update JSON using temporary file
                jq --arg newpath "$new_imagepath" '.imagePath = $newpath' "$dest_dir/$new_filename" > "$dest_dir/tmp.json"
                mv -f "$dest_dir/tmp.json" "$dest_dir/$new_filename"
            fi
        done
    done    
}

# Example usage:
# copySeperatesFile2OneHome_for301pxCasesHasTooMuchFolder "/path/to/source" "/path/to/destination/combined_files"   

test $# -lt 2 && echo -e "Usage:\n\t APP src_dir dest_dir " && exit 0

copySeperatesFile2OneHome_for301pxCasesHasTooMuchFolder "$1" "$2"