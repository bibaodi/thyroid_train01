#!/bin/env bash

function copyModelFile(){
	local SRC=${1:-noparam}
	local DST=${2:-noparam}
	#echo "Debug:${SRC}, ${DST}"
	test "noparam" == ${SRC} && echo "Model Path Is Neededd" && return -51
	test "noparam" == ${DST} && echo "Model baseName Is Neededd" && return -52
	#runs/classify/train32/weights/
	local srcModel="${SRC}/weights/best.pt"
	test ! -f ${srcModel} && echo "$0 :Model file notFound" && return -53
	CMD="cp ${srcModel} ./${DST}.pt"

	eval ${CMD}
	echo ">>>01]eval:$0 $@ \"{CMD}\" ...done"
	return 0
}

function createCrc(){
	md5sum ./model_* >> checksum
	echo ">>>02]$0 $@ ...done"
}

function cpTrainMetrics(){
	local SRC=${1:-noparam}
	#echo "Debug:${SRC} "
	test "noparam" == ${SRC} && echo "$0 :train Path Is Needed" && return -61
	
	local DST="trainMetricsResults"
	mkdir -p ./${DST}
	local files="confusion_matrix* args.yaml results.csv results.png *_batch*.*"
	for ifile in ${files[@]}; do
		#echo "Debug:will copy ${ifile}"
		cp ${SRC}/${ifile} ./${DST}/
	done	

	echo ">>>03]$0 $@ ...done"
}
function createReadMe(){
	local lastModelDir=($(ls -1dr ../model_* |grep -v '.zip'))
	lastModelDir=${lastModelDir[1]}
	echo "Debug:lastModelDir=${lastModelDir}"
	test ! -d ${lastModelDir} && echo "failed get last model folder, please manual do it." && return -71
	cp ${lastModelDir}/readme.md ./

	echo ">>>04]$0 $@ ...done"
}

function main01CopyFiles(){
if test $# -lt 1; then
	echo -e "Usage:\n\t$0 train-result-dir"
	exit 0;
fi
trainResDir=${1:-noTrain}
test ! -d ${trainResDir} && echo "train result folder notFound" && exit 0

curRealPath=`pwd`
curDirName=`basename ${curRealPath}`
prefix="model_"

echo "Debug:${curDirName}"

if [[ ${curDirName} != "$prefix"* ]]; then
	echo "The app should execute in [Model Pacakge Dir] which start with ${prefix}"
	exit 0
fi

copyModelFile ${trainResDir} ${curDirName}
createCrc;
cpTrainMetrics ${trainResDir} ;
createReadMe;
}

function folder2zip(){
	local SRC=${1:-noparam}
	#echo "Debug:${SRC} "
	test "noparam" == ${SRC} && echo "$0 :zip folder Path Is Needed" && return -61
	test ! -d ${SRC} && echo "[${SRC}]folder notFound" && return -62
	SRC=$(realpath ${SRC})
	local bnSRC=$(basename ${SRC})
	local zipfile="${bnSRC}.zip"
	zip ./${zipfile} -r ./${bnSRC} 

	echo ">>>05]$0 $@ ...done"
}

firstParam=${1:-no1stParam}

if test  ${firstParam} == "zip"; then
	folder2zip $2;
	exit 0;
else
	main01CopyFiles $@;
fi

