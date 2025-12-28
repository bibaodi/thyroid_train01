#!/bin/bash

LOGFILE="print.log"

function run_segtrain {
    echo start `date "+%Y-%m-%d %H:%M:%S"`
    local SRCDIR=$1
    local DATADIR=$2
    local PURPOSE=$4 
    local FrmSize=96
    #local KEYLABEL="Interscalene"
    local KEYLABEL="BP"
    local RESDIR=res_${PURPOSE}_${DATADIR}_`date "+%Y%m%dT%H%M"`_sz${FrmSize}

    echo "training ${DATADIR}" `date  "+%Y-%m-%d %H:%M:%S"` | tee $LOGFILE 
    echo "$5" "$6" | tee -a $LOGFILE 
    
    mkdir $RESDIR

    python scripts/train_segmentation_bp.py --datapath /train/data/$DATADIR/np_data --outdir $RESDIR --dataset multi --size ${FrmSize} --combine_train_valid --flip_data --use_augmentation --keylabel $KEYLABEL $5 $6 
  

    cp /train/data/$DATADIR/np_data/*.txt $RESDIR
    #cp run.out $RESDIR
    mv $RESDIR /train/history_train/
    echo end `date "+%Y-%m-%d %H:%M:%S"` | tee -a $LOGFILE 
    mv *.log $RESDIR

}


Method="preprocess"
python setup.py install 

export CUDA_VISIBLE_DEVICES=0,1

Purpose='show'

echo ${Purpose} '|  | without augmentation'
DATA='mul_bpV1.9_191022'
#run_segtrain InterscaleneV2.0.0129 InterscaleneV2.0_0129  $Method $Purpose
run_segtrain $DATA $DATA 'trainonly' $Purpose

