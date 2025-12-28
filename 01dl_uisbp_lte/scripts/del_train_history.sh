#!/bin/bash

ALL_FILES=` du -h /train/history_train/ |grep G |cut -d "G" -f 2`

echo "$ALL_FILES"

for FILE in ${ALL_FILES} ; do 
	echo "to be delete:"${FILE}
	rm -r	${FILE}
done
