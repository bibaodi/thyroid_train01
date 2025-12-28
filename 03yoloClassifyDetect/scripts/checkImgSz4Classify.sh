#!/bin/env bash
#eton@250309 first edition;

function findImagefiles() {
	find $1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" \)

}

rubbish=' /tmp/train-mal'

function getImgWH(){
	local imgfile=${1:-notexist}
	test "notexist" == ${imgfile} && echo "image file is necessary" && return -1

	if ! test -f ${imgfile} ; then
	       echo "image file not exist:[${imgfile}]" && return -1
	fi	       
	dimensions=$(identify -format "%wx%h" ${imgfile})
	local width=${dimensions%x*}
	local height=${dimensions#*x}
	
	edgeThreshold=64
	if test ${width} -lt ${edgeThreshold}  || test ${height} -lt ${edgeThreshold} ; then
	       echo "image file edge[${width}x${height}] less then {edgeThreshold}:[${imgfile}] will be removed" ;
	       mv ${imgfile} ${rubbish} ;
	fi
	if test ${width} -ge $((2 * ${height}))  || test ${height} -ge $((2 * ${width})) ; then
	       echo "image file edge[${width}x${height}] is twice then another:[${imgfile}] will be removed" ;
	       mv ${imgfile} ${rubbish} ;
	fi
	#echo "Width: $width, Height: $height"
	#echo "WH:[$width,$height]"
}

getImgWH $1

export -f getImgWH



function getImgsSizeInOneFolder() {
	local _folder=${1:-notexist}
	test "notexist" == ${_folder} && echo "folder parameter is necessary" && return -1
	if ! test -d ${_folder} ; then
	       echo "folder not exist" && return -1
	fi	       

	#for iimg in `find ${_folder}`; do
	for iimg in `find ${_folder} -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" \)`; do
		getImgWH ${iimg}
	done
	#find ${_folder} | xargs -I {} getImgWH '{}'
	#find ${_folder} | xargs -I {} bash -c 'getImgWH "$@"' _ {}
	#CMD="find ${_folder} | xargs -I {} bash -c 'getImgWH {}'"
	#echo "CMD=${CMD}" && eval $CMD
}

getImgsSizeInOneFolder $1;

echo "${0} end..."
