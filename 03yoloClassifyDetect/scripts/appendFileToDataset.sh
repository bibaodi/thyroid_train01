#!/bin/env bash

function copy2here(){
	_home="../301px-250301-dong5k.yoloBoM/"
	for i in `cat ../301px-250301-dong5k.yoloBoM/val-dong5k.list`; do 
		img="${_home}/images/$i"
		CMD="cp ${img} ./images/val/"
		echo "CMD=${CMD}"
		eval ${CMD}
		label=${i/.jpg/.txt}
		label="${_home}/labels/$label"
		CMD="cp ${label} ./labels/val/"

		echo "CMD=${CMD}"
		eval ${CMD}
	done
	echo "DONE....."
}
function delItemPairInList(){
	_home="../301px-250301-dong5k.yoloBoM/"
	for i in `cat ../301px-250301-dong5k.yoloBoM/val-dong5k.list`; do 
		img="${_home}/images/$i"
		CMD="mv ${img} /tmp/"
		echo "CMD=${CMD}"
		eval ${CMD}
		label=${i/.jpg/.txt}
		label="${_home}/labels/$label"
		CMD="mv ${label} /tmp/"

		echo "CMD=${CMD}"
		eval ${CMD}
	done
	echo "DONE....."
}


#copy2here;
delItemPairInList;
