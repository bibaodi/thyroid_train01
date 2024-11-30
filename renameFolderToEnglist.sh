#!/bin/bash
function renameFolderToEng() {
	for ifolder in `find . -maxdepth 1 -type d -not -path .`; do
		local inewName=`sed  's/[^0-9]\+\([0-9]\{1,\}\)[^0-9]*/thyroidNodules_axp-\1/g' <<< "${ifolder}"`
		local inewNamePad0=`sed -E 's/([^0-9])([0-9])([^0-9]|$)/00\2\3/g; s/([^0-9])([0-9]{2})([^0-9]|$)/0\2\3/g' <<< "${inewName}"`
		local CMD="mv \"${ifolder}\" \"${inewNamePad0}\""
		echo "CMD=${CMD}"	
		eval ${CMD}
	#sed  's/[^0-9]\+\([0-9]\{1,\}\)[^0-9]*/thyroidNodules_axp-\1/g'|sed -E 's/([^0-9])([0-9])([^0-9]|$)/00\2\3/g; s/([^0-9])([0-9]{2})([^0-9]|$)/0\2\3/g'
done
}

renameFolderToEng;

