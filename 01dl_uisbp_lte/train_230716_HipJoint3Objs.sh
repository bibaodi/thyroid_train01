#!/bin/bash
LOGFILE="train_ddh3objs.log"
function run_segtrain {
    echo start `date "+%Y-%m-%d %H:%M:%S"`
    local SRCDIR=$1
    local DATADIR=$2
    local PURPOSE=$4 
    local FrmSize=96
    #local FrmSize=224
    #local KEYLABEL="Interscalene"
    local KEYLABEL="Labrum"
    local KEYLABEL="IliacLower"
    #local KEYLABEL="HipJoint"
    local RESDIR=res_${PURPOSE}_${DATADIR}_`date "+%Y%m%dT%H%M"`_sz${FrmSize}

    echo "training ${DATADIR}" `date  "+%Y-%m-%d %H:%M:%S"` | tee $LOGFILE 
    echo "$5" "$6" | tee -a $LOGFILE 
    echo "result directory=${RESDIR}" | tee -a $LOGFILE 
    
    mkdir $RESDIR

    python scripts/train_DDH_3objs.py --datapath /train/data/$DATADIR/np_data \
	    --outdir $RESDIR --dataset multi --size ${FrmSize} \
	    --combine_train_valid    \
	    --keylabel $KEYLABEL $5 $6 | tee -a $LOGFILE 
	    #--checkpoint  /train/history_train/res_Labrum4DDH2Objs230326_mul_HipJointLabrum2OBJ1.0_N5R2_20230326T2207_sz96/linknet18_32_0.5_multi_96/weights.h5 \ 
	    #--combine_train_valid  --checkpoint  /train/history_train/res_Labrum4DDH2Objs230326_mul_HipJointLabrum2OBJ1.0_N5R2_20230326T2207_sz96/linknet18_32_0.5_multi_96/weights.h5 
#--use_augmentation 
 	trainRet=$? 
    # mkdir $RESDIR
    cp /train/data/$DATADIR/np_data/*.txt $RESDIR
    #cp run.out $RESDIR
    echo end `date "+%Y-%m-%d %H:%M:%S"` | tee -a $LOGFILE 
    mv *.log $RESDIR/
    mv $RESDIR /train/history_train/
    if [ ${trainRet} -eq 0 ] ; then
    echo " send it to S3 >>${RESDIR}"
    aws s3 cp /train/history_train/${RESDIR} s3://ftp-huaxia-200420/train_history/${RESDIR} --recursive
	else
		echo "not correct finish."
	fi
}


Method="preprocess"
python setup.py install 

export CUDA_VISIBLE_DEVICES=0,1

Purpose='DDH3rdObjs'

echo ${Purpose} '|no-augmentation'

DATA='mul_HipJoint3OBJ1.0_N38'
DATA='mul_HipJointLabrum2OBJ1.0_N38'
DATA='mul_HipJointLabrum2OBJ1.0_N3Twin'
DATA='mul_HipJointLabrum2OBJ1.0_N5R2'
DATA='mul_HipJoint3OBJV0.1_N8'
DATA='mul_HipJoint3OBJV0.1_N18'
DATA='mul_HipJoint3OBJV0.1_N37'
DATA='mul_HipJoint3OBJV0.1_N19s'
DATA='mul_HipJoint3OBJV0.1_N19sPurgeIL'
DATA='mul_HipJoint3OBJV0.1_N5S224'
DATA='mul_HipJoint3OBJV0.2_N5Eton'

#DATA='mul_plaque7.0paddingblack191107'
#run_segtrain InterscaleneV2.0.0129 InterscaleneV2.0_0129  $Method $Purpose
run_segtrain $DATA $DATA 'trainonly' $Purpose


echo "will shutdown it in 10 seconds. you can input any character to ignore it."
read -t 10 -n 1 ans
if [ $? == 0 ]; then
    echo "Your answer is: $ans"
else
    echo "Can't wait anymore! shutdown..."
    sudo shutdown -h now
fi

#end
