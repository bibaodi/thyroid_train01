#!/bin/bash
#/221024
#aws s3 cp s3://ftp-huaxia-200420/usData-FormalRelease/10lastVersion-LocalAll/221024_HipJointsForDDH/221024_DDH_labelmeFormat.zip /tmp/
#cp /tmp/221024_DDH_labelmeFormat.zip /data/raw_data/
FNAME="HipJoint_DDH_N38_V1.0.zip"
CATEGORY="HipJoint"
#/221112
#CATEGORY="belowAbdominalFat"
FNAME="fats_belowAbdominalN20_221111.zip"
FNAME="DDH_SpoonLips_iliumdestinationDatas221211.zip"
SRC="s3://ftp-huaxia-200420/usData-FormalRelease/20export-FormalSend/${CATEGORY}/${FNAME}"
SRC="s3://ftp-huaxia-200420/usData-FormalRelease/20export-FormalSend/HipJoint/21-modifiedByEton.zip"

# run it.
CMD="aws s3 cp ${SRC}  /data/raw_data/"
echo "run CMD=${CMD} y/n ?"
read ans
if [ ${ans} != 'Y' -a ${ans} != 'y' ] ; then
	echo "you choose Not y/Y."
else
	eval ${CMD}
fi

echo "Finish"


