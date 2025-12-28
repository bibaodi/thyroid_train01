#!/bin/bash
##eton@250116 add not cropped images to data set.

function move_top_n(){
# Source and destination directories
SOURCE_DIR="${1:-/path/to/source}"
DEST_DIR="${2:-/path/to/destination}"

test $# -lt 2 && echo "[$#]Param Not Enough [App In Out]" && return 0
test ! -d ${SOURCE_DIR} && echo "Source [${SOURCE_DIR}] not exist" && return 0
test ! -d ${DEST_DIR} && echo "DEST [${DEST_DIR}] not exist" && return 0

# Create the destination directory if it doesn't exist
#mkdir -p "$DEST_DIR"

# Move the top 100 items
ls "$SOURCE_DIR" | head -n 100 | while read file; do
  CMD="mv $SOURCE_DIR/$file $DEST_DIR"
  echo "CMD=${CMD}"  && eval ${CMD}
done
}

move_top_n $1 $2;


function addNotCroppedImgsToDataset(){
	echo "function:"		

}

