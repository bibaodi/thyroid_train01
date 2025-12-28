#!/bin/bash
LOGFILE="train_ddh3objs.log"
function run_segtrain {
    echo start `date "+%Y-%m-%d %H:%M:%S"`
    local SRCDIR=$1
    local DATADIR=$2
    local PURPOSE=$4 
    local FrmSize=96
    #local KEYLABEL="Interscalene"
    local KEYLABEL="HipJoint"
    local RESDIR=res_${PURPOSE}_${DATADIR}_`date "+%Y%m%dT%H%M"`_sz${FrmSize}

    echo "training ${DATADIR}" `date  "+%Y-%m-%d %H:%M:%S"` | tee $LOGFILE 
    echo "$5" "$6" | tee -a $LOGFILE 
    echo "result directory=${RESDIR}" | tee -a $LOGFILE 
    
    mkdir $RESDIR

    python scripts/train_segmentation_thyroid_tf2.py --datapath /train/data/$DATADIR/np_data \
	    --outdir $RESDIR --dataset multi --size ${FrmSize} \
	    --combine_train_valid    \
	    --keylabel $KEYLABEL $5 $6 | tee -a $LOGFILE 
	    #--combine_train_valid  --checkpoint  /train/history_train/res_tryDDH_mul_HipJoint1.0_N38_20221102T2058_sz96/linknet18_32_0.5_multi_96/weights.h5 \
#--use_augmentation 
  
    # mkdir $RESDIR
    cp /train/data/$DATADIR/np_data/*.txt $RESDIR
    #cp run.out $RESDIR
    echo end `date "+%Y-%m-%d %H:%M:%S"` | tee -a $LOGFILE 
    mv *.log $RESDIR/
    mv $RESDIR /train/history_train/
}


Method="preprocess"
python setup.py install 

export CUDA_VISIBLE_DEVICES=0,1

Purpose='DDH3Objs'

echo ${Purpose} '|no-augmentation'

DATA='mul_HipJoint3OBJ1.0_N38'

#DATA='mul_plaque7.0paddingblack191107'
#run_segtrain InterscaleneV2.0.0129 InterscaleneV2.0_0129  $Method $Purpose
run_segtrain $DATA $DATA 'trainonly' $Purpose

echo "will shutdown it."

sudo shutdown -h now

