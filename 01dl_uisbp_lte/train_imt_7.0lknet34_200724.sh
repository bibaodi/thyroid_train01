#!/bin/bash
LOGFILE="train_im.log"
function run_segtrain {
    echo start `date "+%Y-%m-%d %H:%M:%S"`
    local SRCDIR=$1
    local DATADIR=$2
    local PURPOSE=$4 
    local FrmSize=96
    #local KEYLABEL="Interscalene"
    local KEYLABEL="IM"
    local RESDIR=res_${PURPOSE}_${DATADIR}_`date "+%Y%m%dT%H%M"`_sz${FrmSize}

    echo "training ${DATADIR}" `date  "+%Y-%m-%d %H:%M:%S"` | tee $LOGFILE 
    echo "$5" "$6" | tee -a $LOGFILE 
    
    mkdir $RESDIR

    python scripts/train_segmentation_ime_tf2.py --datapath /train/data/$DATADIR/np_data --model linknet34 \
	    --outdir $RESDIR --dataset multi --size ${FrmSize} \
	    --combine_train_valid  --batch_size 256 \
	    --checkpoint /train/history_train/res_lkn34-2_mul_imt7.4above28_448_20200724T1445_sz96/linknet34_32_0.5_multi_96/weights.h5 \
	    --keylabel $KEYLABEL $5 $6 | tee -a $LOGFILE 
  #--use_augmentation 

    # mkdir $RESDIR--use_augmentation 
    cp /train/data/$DATADIR/np_data/*.txt $RESDIR
    #cp run.out $RESDIR
    echo end `date "+%Y-%m-%d %H:%M:%S"` | tee -a $LOGFILE 
    mv *.log $RESDIR/
    mv $RESDIR /train/history_train/
}


Method="preprocess"
python setup.py install 

export CUDA_VISIBLE_DEVICES=0,1

Purpose='lkn34-5'

echo ${Purpose} '|augmentation'

DATA='mul_imt7.0_224'
#DATA='mul_imt7.4above28_448'
#DATA='mul_plaque7.0paddingblack191107'
#run_segtrain InterscaleneV2.0.0129 InterscaleneV2.0_0129  $Method $Purpose
run_segtrain $DATA $DATA 'trainonly' $Purpose

