#!/bin/bash
dcms=`find ./ -name "*.dcm"`
for dcm in $dcms; 
do
	rm $dcm
	echo "delete dcm"$dcm
done
