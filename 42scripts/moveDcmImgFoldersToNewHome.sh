#!/bin/bash

function moveDcmImgFoldersToNewHome() {
for ifolder in `find ./ -type d -name "*.dcm_frms"`; 
	do 
		mv ${ifolder} ../241208-incremental4cases/ ; 
	
	done
}
