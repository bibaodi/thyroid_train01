#!/bin/bash
S="s3://ftp-huaxia-200420/usData-FormalRelease/20export-FormalSend/HipJoint/230222-LabrumHipjointOnly.zip"
ddir="/data/raw_data/HipJointV02/"
CMD="aws s3 cp ${S} ${ddir}"
echo -e "CMD=$CMD\n run?y/n"

read ans

if [ 'y' == $ans ] ; then
	eval $CMD
else
	echo "y is needed. exit"
fi

# 2-unzip 
echo "02: unzip file."

# 3-process data using
echo "03:process_ddh2Obj_data.sh, \n\t /data/mul_HipJointLabrum2OBJ1.0_N38/"

