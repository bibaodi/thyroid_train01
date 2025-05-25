#!/bin/bash

function moveDcmImgFoldersToNewHome() {
for ifolder in `find ./ -type d -name "*.dcm_frms"`; 
	do 
		cp -r ${ifolder} ../241208-incremental4cases/ ; 
	done
}
