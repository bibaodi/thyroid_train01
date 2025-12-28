#!/bin/bash

aws s3 cp s3://ftp-huaxia-200420/train_history/res_tryDDH_mul_HipJoint1.0_N38_20221026T1006_sz96/ ./ --recursive
aws s3 cp ./HipJoint_DDH_N38_V2.0_recheckAll.zip s3://ftp-huaxia-200420/usData-FormalRelease/20export-FormalSend/HipJoint/
aws s3 cp  s3://ftp-huaxia-200420/usData-FormalRelease/20export-FormalSend/HipJoint/230226_labrumFixWithHip_N3.zip  /data/raw_data/HipJointV02/ 
