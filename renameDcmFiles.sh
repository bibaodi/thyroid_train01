#/bin/bash

function renameAllDcmfile() {
	for idcm in `find ./ -type f ! -name '*.json'  -size +10M` ;
		do 
			fullPath=`realpath "${idcm}"`
			baseName=`basename "${fullPath}"`
			dirName=`dirname "${fullPath}"`
			#echo "debug: ${fullPath} ; ${baseName}"
			inewDcm=`echo ${baseName}|sed  's/[^0-9]\+\([0-9]\{1,\}\)/thyroidNodules_axp-\1.dcm/g'`;
		        inewDcm=`sed -E 's/([^0-9])([0-9])([^0-9]|$)/00\2\3/g; s/([^0-9])([0-9]{2})([^0-9]|$)/0\2\3/g' <<<"${inewDcm}"`	
			infDcm="${dirName}/${inewDcm}"
			#echo "${idcm}-->${inewDcm} -->> ${infDcm}"; 
			#echo "-->> ${infDcm}"; 
			CMD="mv ${fullPath} ${infDcm}"
			local echo "CMD=${CMD}"
			eval ${CMD}
		done
}

renameAllDcmfile;

