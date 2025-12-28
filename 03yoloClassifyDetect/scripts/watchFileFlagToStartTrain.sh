#!/bin/bash

_ff="/tmp/ti.done"
NN=0
while true ; do
	echo "Hii:$((NN+=1))"
	if test -f "/tmp/ti.done" ; then
		echo "file here." && rm ${_ff} 
		##conda activate detectron2
		python ./trainBenignMalign_v04.py
		break;
	fi
	sleep 60;
done
