#!/bin/bash

# Directory path and number of files to delete
directory="/path/to/your/directory"
directory=${1:-'nowhere'}
num_files=${2:-1}

MSG="Deleted $num_files files from $directory"

	read -p "${MSG}? y/n"$'\n' ans
	test ${ans} != 'y' && exit 0

# List files, shuffle, and delete the specified number of files
ls -1 "$directory" | shuf | head -n "$num_files" | xargs -I {} rm -- "$directory/{}"

echo "${MSG} ...Done"
