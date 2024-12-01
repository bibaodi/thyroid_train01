#/bin/bash
##     1. dcm文件重命名'甲结节1' -> 'thyroidNodules-axp-001.dcm';
function renameAllDcmfile() {
	for idcm in `find ./ -type f ! -name '*.json'  -size +2M` ;
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
			local CMD="mv ${fullPath} ${infDcm}"
			echo "CMD=${CMD}"
			read -p "confirm Do It? y/n "$'\n' ans
			test ${ans} == 'y' && echo "do it" && eval ${CMD}
			#eval ${CMD}
		done
}

function renameDcmRemoveDuplicate(){
	yesForAll=0
	for idcm in `find ./ -type f  -name '*.dcm'  -size +2M`;
		do 
			indcm2=`sed 's/\.dcm\.dcm*/.dcm/' <<< "${idcm}"`
			
			local CMD="mv ${idcm} ${indcm2}"
			echo "CMD=${CMD}" 
			if test ${yesForAll} -eq 0 ; then
				read -p "confirm Do It? y/n "$'\n' ans
				test ${ans} == 'y' && echo "do it" && eval ${CMD}
				test ${ans} == 'A' && echo "yes for all." && yesForAll=1 && eval ${CMD}
			else
				eval ${CMD}
			fi
		done

}

#renameAllDcmfile;

renameDcmRemoveDuplicate;
