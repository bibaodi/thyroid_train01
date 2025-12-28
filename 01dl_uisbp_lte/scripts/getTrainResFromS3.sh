#!/bin/bash

function getTrainHistoryFolder(){
	#res_tryThyNod_axpG31Only_mul_nodules1.3_axp31_224_20241203T1937_sz224
	local trainRes=${1:-trainResult}
	local CMD="aws s3 cp  s3://ftp-huaxia-200420/train_history/${trainRes}/ ./ --recursive"
	read -p "run CMD=${CMD} y/n"$'\n' res
	test 'y' == $res && eval ${CMD}
}

function showAllTrainRes(){
	 aws s3 ls s3://ftp-huaxia-200420/train_history/  --human-readable --summarize
}

if test $# -lt 1 ; then
       	showAllTrainRes;
else 
	getTrainHistoryFolder "$1"
fi
