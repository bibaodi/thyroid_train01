#!/bin/bash


function renameImgfilesInfolder2dirPlusImg(){
# Target directory
target_dir="${1:-LTLLON}"

echo "target_dir=${target_dir}"

# Process all JSON files in subdirectories
find "$target_dir" -type f -path "*_frms/frm-*.png" -print0 | while IFS= read -r -d $'\0' file; do
    # Extract parent directory name and filename
    parent_dir=$(dirname "$file")
    base_name=$(basename "$parent_dir")
    file_name=$(basename "$file")
    
    # Construct new filename by merging directory and filename
    new_name="${base_name}_${file_name}"
    
    # Move file to target directory with new name
    CMD="mv -v \"$file\" \"$target_dir/$new_name\""
    echo "CMD=$CMD"
    eval ${CMD}
done

# Remove empty _frms directories (optional)
find "$target_dir" -type d -name "*_frms" -empty -delete
}

for icls in `ls`; do
	#renameImgfilesInfolder2dirPlusImg './LTLLON'
	renameImgfilesInfolder2dirPlusImg "${icls}"
done
