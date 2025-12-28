#!/bin/bash
function run_segtrain {
    echo start `date "+%Y-%m-%d %H:%M:%S"`
    local SRCDIR=$1
    local DATADIR=$2
    local PURPOSE=$4 
    local FrmSize=96
    #local KEYLABEL="Interscalene"
    local KEYLABEL="Plaque"
    local RESDIR=res_${PURPOSE}_${DATADIR}_`date "+%Y%m%dT%H%M"`_sz${FrmSize}

    if [ "$3" == "preprocess" ]
       then
         echo "preprocess data ${DATADIR}"
         python scripts/process_data.py --dataset multi  --rootdir ~/train/$SRCDIR --outdir /data/$DATADIR --keylabel $KEYLABEL
    fi

    echo "training ${DATADIR}" `date  "+%Y-%m-%d %H:%M:%S"`
    echo "$5" "$6"
    
    mkdir $RESDIR

    python scripts/train_segmentation.py --datapath /data/$DATADIR/np_data --outdir $RESDIR --dataset multi --size ${FrmSize} --combine_train_valid --keylabel $KEYLABEL $5 $6 > ${PURPOSE}_${DATADIR}_ln18_${FrmSize}.log 2>&1
  
    # mkdir $RESDIR
    mv *.log $RESDIR
    cp /data/$DATADIR/np_data/*.txt $RESDIR
    echo end `date "+%Y-%m-%d %H:%M:%S"`
    cp run.out $RESDIR
}


Method="trainonly"
if [ "$1" == "preprocess" ]
  then
    Method="preprocess"
fi

echo 'activate uisbp env'
source activate uisbp

python setup.py install > /dev/null

export CUDA_VISIBLE_DEVICES=0,1

Purpose='Plaque.V6.0'

echo ${Purpose} '| emptyScanV2 | without augmentation'
DATA='mul_plaque_emptyscan.v6.0_0806'
#run_segtrain InterscaleneV2.0.0129 InterscaleneV2.0_0129  $Method $Purpose
run_segtrain $DATA $DATA 'trainonly' $Purpose

