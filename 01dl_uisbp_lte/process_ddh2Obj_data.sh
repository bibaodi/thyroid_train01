#!/bin/bash
## for DDH data train.
echo "usage: app + rootdir + output_dir + 96|224|336|448"
echo "root dir is: "$1
echo "output dir is: "$2
#--------
abandonlist="/train/data/raw_data/abandoned_video_list_190927.txt"
IND="/data/raw_data/HipJoint/HipJoint_N38_221024"
IND="/data/raw_data/HipJointV02"
IND="/data/raw_data/DDH_SpoonLips_iliumdestinationDatasV1-221211"
IND="/data/raw_data/HipJointV02/230222-LabrumHipjointOnly"
IND="/data/raw_data/HipJointV02/N3"
IND="/data/raw_data/HipJointV02/230326-N5LabrumWithHipJoint"
IND="/data/raw_data/HipJointV02/230401-N38RefinedLabrmDDH"
OUTD="/data/mul_HipJoint3OBJ1.0_N38"
OUTD="/data/mul_HipJointLabrum2OBJ1.0_N38"
OUTD="/data/mul_HipJointLabrum2OBJ1.0_N3Twin"
OUTD="/data/mul_HipJointLabrum2OBJ1.0_N5R2"
OUTD="/data/mul_HipJointLabrum2OBJ_N38"
SIZE=96
KEYL="HipJoint"
KEYL="Labrum"

#CMD="python scripts/process_data.py --rootdir $IND --dataset multi --outdir $OUTD --imagesize $SIZE --keylabel $KEYL --abandon $abandonlist --black_it"

CMD="python scripts/process_data.py --rootdir $IND --dataset multi --outdir $OUTD --imagesize $SIZE --keylabel $KEYL --abandon $abandonlist"
echo $CMD

$CMD
