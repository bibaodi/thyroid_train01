# show class_id in yolo labels.txt dataset
# eton@250315 v0.2;
##eton@250315 add '-L' to find, otherwise the type f not work for symbolic link; 
getYoloLabelsFirstNumAndCheck(){

local _targetVal='0' #0,1,...100

local labelDir=${1:-nodir}

test ${labelDir} == 'nodir' && echo "usage:APP label-dir" && return 0;
test ! -d ${labelDir} && echo "not found:${labelDir}" && return 0;

local awk_cmdStr='{  if ($1 != awk_var) { \
     print FILENAME ":line" NR ": " $0 ",target:" awk_var \
   }\
}'


CMD="find -L ${labelDir} -type f -iname \"*.txt\" -print0"
echo "CMD=$CMD"

find -L ${labelDir} -type f -iname "*.txt" -print0 | xargs -0 awk -v awk_var="$_targetVal" "${awk_cmdStr}" 

echo "run [$0 $@] done..."
}



getYoloLabelsFirstNumAndCheck $1;
