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
IND="/data/raw_data/HipJoint3Labels/DDH-3lablesIliacLowerHipjointLabrumV1_230531"
IND="/data/raw_data/HipJoint3Labels/230624-afterMeetingKeepSmoth"
IND="/data/raw_data/HipJoint3Labels/only5ddh3objs"
IND="/data/raw_data/HipJoint3Labels/230708-etonModified3lableType"
OUTD="/data/mul_HipJoint3OBJV0.1_N18"
OUTD="/data/mul_HipJoint3OBJV0.1_N37"
OUTD="/data/mul_HipJoint3OBJV0.1_N19sPurgeIL"
OUTD="/data/mul_HipJoint3OBJV0.1_N5S224"
OUTD="/data/mul_HipJoint3OBJV0.2_N5Eton"
labels="['IliacLower', 'Labrum', 'HipJoint']"
SIZE=96
#SIZE=224
KEYL="IliacLower"
#KEYL="Labrum"

#CMD="python scripts/process_data.py --rootdir $IND --dataset multi --outdir $OUTD --imagesize $SIZE --keylabel $KEYL --abandon $abandonlist --black_it"

CMD="python scripts/process_data.py --rootdir $IND --dataset multi --outdir $OUTD --imagesize $SIZE --keylabel $KEYL --abandon $abandonlist"
echo $CMD

$CMD


echo "END./"
