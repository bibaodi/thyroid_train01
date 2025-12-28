#!/bin/bash

abandonlist="/train/data/raw_data/abandoned_video_list_190927.txt"
IND=$1
OUTD=$2
SIZE=$3
KEYL="BP"

CMD="python scripts/process_data.py --rootdir $IND --dataset multi --outdir $OUTD --imagesize $SIZE --keylabel $KEYL --abandon $abandonlist --black_it"
echo $CMD

$CMD
