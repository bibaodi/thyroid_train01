#!/bin/bash

echo "usage: app + rootdir + output_dir + 96|224|336|448"
echo "root dir is: "$1
echo "output dir is: "$2
#--------
abandonlist="/train/data/raw_data/abandoned_video_list_190927.txt"
IND=$1
OUTD=$2
SIZE=$3
KEYL="Thyroid"

#CMD="python scripts/process_data.py --rootdir $IND --dataset multi --outdir $OUTD --imagesize $SIZE --keylabel $KEYL --abandon $abandonlist --black_it"

CMD="python scripts/process_data.py --rootdir $IND --dataset multi --outdir $OUTD --imagesize $SIZE --keylabel $KEYL --abandon $abandonlist"
echo $CMD

$CMD
