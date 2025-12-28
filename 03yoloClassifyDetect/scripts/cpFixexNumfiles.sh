#!/bin/bash

# Directory path and number of files to delete
directory="/path/to/your/directory"
directory=${1:-'nowhere'}
num_files=${2:-1}
fromdir=${3:-'nowhere'}

if test $# -lt 3; then
	echo "Usage:$0 todir Num fromdir"	
	exit 0
fi
MSG="cp $num_files files from ${fromdir} to  $directory ?"

	read -p "${MSG}? y/n"$'\n' ans
	test ${ans} != 'y' && exit 0

# List files, shuffle, and delete the specified number of files
ls -1 "${fromdir}" | shuf | head -n "${num_files}" | xargs -I {} cp "${fromdir}/{}" "$directory/"

echo "${MSG} ...Done"
