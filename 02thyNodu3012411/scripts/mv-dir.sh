#!/bin/bash
dcms=`find ./ -name "*-v*"`
for dcm in $dcms; 
do
	subs=`find ./$dcm -type d -print0 |xargs -0 mv -t ./`
	echo "find in "$dcm">>>"$subs
	rmdir $dcm
done
