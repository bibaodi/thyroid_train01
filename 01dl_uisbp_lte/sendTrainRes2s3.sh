#!/bin/bash
#Dir='res_converge_mul_plaque9.0_20200429T0649_sz96'

Dir=$1

echo -e "will trans \n[$1]\n y/n?"

read answer 
if [ $answer != 'y' ]; then
	echo "choose not Y; return"
	exit 0
fi
aws s3 cp /train/history_train/$Dir s3://ftp-huaxia-200420/train_history/$Dir --recursive

echo "copy $Dir to S3 finish..."
