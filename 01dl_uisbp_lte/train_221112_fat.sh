#!/bin/bash
DateStr=$(date "+%Y%m%dT%H%M%S")
LOGFILE="train_lowerAbdominalFat${DateStr}.log"

echo "log file=${LOGFILE}"

function run_segtrain {
    echo start `date "+%Y-%m-%d %H:%M:%S"`
    local SRCDIR=$1
    local DATADIR=$2
    local PURPOSE=$4 
    local FrmSize=96
    local KEYLABEL="LowerAbdominalFat"
    local RESDIR=res_${PURPOSE}_${DATADIR}_${DateStr}_sz${FrmSize}

    echo "training ${DATADIR}" `date  "+%Y-%m-%d %H:%M:%S"` | tee $LOGFILE 
    echo "$5" "$6" | tee -a $LOGFILE 
    echo "result directory=${RESDIR}" | tee -a $LOGFILE 
    
    mkdir $RESDIR

    python scripts/train_segmentation_thyroid_tf2.py --datapath /train/data/$DATADIR/np_data \
	    --outdir $RESDIR --dataset multi --size ${FrmSize} \
	    --combine_train_valid  \
	    --keylabel $KEYLABEL $5 $6 | tee -a $LOGFILE 
#--use_augmentation 
  
    # mkdir $RESDIR
    cp /train/data/$DATADIR/np_data/*.txt $RESDIR
    #cp run.out $RESDIR
    echo end `date "+%Y-%m-%d %H:%M:%S"` | tee -a $LOGFILE 
    mv *.log $RESDIR/
    mv $RESDIR /train/history_train/
    echo " send it to S3 >>${RESDIR}"
    aws s3 cp /train/history_train/${RESDIR} s3://ftp-huaxia-200420/train_history/${RESDIR} --recursive
    # shutdown machine
    sudo shutdown -h now
}


Method="preprocess"
python setup.py install 

export CUDA_VISIBLE_DEVICES=0,1

Purpose='tryFat'

echo ${Purpose} '|no-augmentation'

DATA='mul_belowAbdominalFatN20'

#DATA='mul_plaque7.0paddingblack191107'
#run_segtrain InterscaleneV2.0.0129 InterscaleneV2.0_0129  $Method $Purpose
run_segtrain $DATA $DATA 'trainonly' $Purpose

