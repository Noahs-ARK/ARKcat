#!/bin/bash

DATASET=~/datasets/sst2/
W2V_LOC=~/datasets/output.txt
SAVE_LOC=../output/foo
MODEL_FILE=~/datasets/'rand_reg_cnn_30.models'
curr_procs=0
NUM_PROCS=5

echo "about to run.py"
# python run.py $DATASET $W2V_LOC $SAVE_LOC 1 -f $MODEL_FILE 1 >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
printf "%s\0" {1..10} | xargs -0 -I @ -P 4 python run.py $DATASET $W2V_LOC $SAVE_LOC 1 -f $MODEL_FILE @ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
# for ((i=0; i<$NUM_PROCS; i++)); do
#   (
#     while read -r line; do
#       python run.py $DATASET $W2V_LOC $SAVE_LOC 1 -f $MODEL_FILE $i >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
#     done < <(awk -v num_procs="$NUM_PROCS" -v i="$i" \
#                  'NR % num_procs == i { print }' <"$MODEL_FILE")
#   ) &
# done
