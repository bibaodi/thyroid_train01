#!/bin/bash
## CMD=mv "./甲结节12_甲状腺结节标注_标注_审核_仲裁" "thyroidNodules_axp012"
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

function copyPatchParsedJsonToTarget(){
	for icase in `find . -type d -not -path .`; do 
		CMD="cp ${icase}/*.json  /mnt/f/241129-zhipu-thyroid-datas/31-labelmeFormatOrganized/231129-thyroidNodulesAix72/${icase}.dcm_frms/"; 
		echo "CMD=$CMD" && eval $CMD; 
	done
}

renameFolderToEng;

