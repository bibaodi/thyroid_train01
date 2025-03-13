#!/bin/bash
## eton@250113 used for remove all err labeles for Yolo Which only has one field but should has 5 fields;
#
function findNofFieldsInTxt(){
	local _f=${1:-notFoundOrParamNotPresent} 
	if test ! -f ${_f} ; then
	       	echo "[${_f}] not exist." && return 0
	fi
	Nfields=`cat ${_f} | awk '{print NF}'`;
	#echo "debug: ${_f} has Fields=${Nfields}" ; 
	test ${Nfields} -lt 5 && echo "Err:${_f}" && return -1
	return 0
}

function testIt(){
	findNofFieldsInTxt ./val/301PACS02-2210290143.02_frm-0001.txt;
}

function findInDir(){
	local _d=${1:-notFoundOrParamNotPresent} 
	if test ! -d ${_d} ; then
	       	echo "[${_d}] not exist." && return 0
	fi
	
	local dustbin="../ErrOnlyOneFieldImgs/"
	mkdir -p ${dustbin}

	for itxt in `find ${_d} -name "*.txt"`;do
	findNofFieldsInTxt ${itxt}
	ret=$?
	if test ${ret} -ne 0 ; then
		CMD="mv ${itxt} ${dustbin}"
		echo "01CMD=${CMD}" && eval ${CMD}
		iimg="${itxt%.txt}.png"
		iimg="../images/${iimg}"
		#file ${iimg} ; 
		CMD="mv ${iimg} ${dustbin}"
		echo "02CMD=${CMD}" && eval ${CMD}
	fi
	done
}

findInDir $1;
## 301pacsDataInLbmfmtRangeY22-24.yoloTiRads1TO6_250113/labels$ bash  findFieldsNotCorrect.sh ./val/
echo "end./"
