#!/bin/bash
function sendThyDataSetToS3(){
	#aws s3 cp ./241208-inc*.zip  s3://ftp-huaxia-200420/usData-FormalRelease/10lastVersion-LocalAll/Thyroid/
	local _zipfile=${1:-notExist}
	test ${_zipfile} = 'notExist' && echo -e "Usage:\n\t APP dataset.zip " && exit 0
	read -p "store ${_zipfile} to S3? y/n"$'\n' ans
	test ${ans} != 'y' && exit 0
	aws s3 cp ${_zipfile} s3://ftp-huaxia-200420/usData-FormalRelease/10lastVersion-LocalAll/Thyroid/
}
echo "use:"
echo "aws s3 cp ./thyroidNodule4BenMalMulti250322.zip s3://ftp-huaxia-200420/usData-FormalRelease/10lastVersion-LocalAll/Thyroid/301PX/"
echo "if you want send to 301PX"

sendThyDataSetToS3 "$1";
