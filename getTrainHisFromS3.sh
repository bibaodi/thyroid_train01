#!/bin/bash
function getTrainHisFromS3(){
#s3 cp s3://ftp-huaxia-200420/train_history/res_optimizeThyNod1.6_mul_nodules1.5_axpG63_224_20241206T0943_sz224 ./res_optimizeThyNod1.6_mul_nodules1.5_axpG63_224_20241206T0943_sz224 --recursive 
local _train=${1:-notExist}
test ${_train} = 'notExist' && echo -e "Usage:\n\t APP trainFolderName " && exit 0
read -p "download ${_train}? y/n"$'\n' ans
test ${ans} != 'y' && exit 0
s3 cp s3://ftp-huaxia-200420/train_history/${_train} ./${_train} --recursive	
}

getTrainHisFromS3 $1;
