#!/bin/bash
function test4CtrlM_in_textfule(){
	iscore=0.999 ; test $(bc -l <<< "${iscore:-0} > 0.99"; ) -eq 1 && echo "yes"
	#exit 0
}

echo "Usage: please run dos2unix if text error.--eton@250517"

function groupImagesByModelResult(){
	local csvName=${1:-nofile}
	test ! -f ${csvName} && echo "not exist csvfile=[${csvName}]" && return -1;

	local clsName=${2:-noclsname}
	test 'noclsname' == ${clsName} && echo "not valid class name=[${clsName}]" && return -2;
	
	local outputFolder="groupBy${clsName}99"
	mkdir -p ${outputFolder}
	test ! -d ${outputFoler} && echo "not exist output folder name=[${outputFolder}]" && return -3
	echo "processing csvfile=[${csvName}], class name=[${clsName}]"
	
	N=0; #IFS=$' \t\n'; 
for iline in `cat ${csvName}`; 
do  
	#echo "debug: line=[${iline}]"
	#read -r iimgname iclass iscore <<< "${iline//,/ }" ;
	read -r iimgname iclass iscore <<< $(sed 's/,/ /g' <<<${iline}) ;
	if ! [[ "$iscore" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    		echo "[$iscore]: is not a number" 
		N=$((N + 1)); 
		test $N -gt 1 && break; 
		continue;
	fi
	test $(bc -l <<< "${iscore:-0} < 0.99"; ) -eq 1 && continue;  
	test ${iclass} != "${clsName}" && continue;

	echo "name: ${iimgname}; score=[${iscore}]; class=[$iclass]"; 

	find '../250321excels__Xp23!@79dT/' -name "${iimgname}*"  -exec cp {} ${outputFolder} \;

	N=$((N + 1));
	echo "countN=${N}"	
	#test $N -gt 9 && break; 
done
}

groupImagesByModelResult  glandPosResultsSingleNodu250514.csv "LTLTRA" 
groupImagesByModelResult   glandPosResultsSingleNodu250514.csv "LTLTRA" 
