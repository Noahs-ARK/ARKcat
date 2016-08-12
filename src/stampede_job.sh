#!/bin/bash

INPUT_DIR='/work/04256/tg835555'
# INPUT_DIR='/home/katya/datasets'
DATASET=$INPUT_DIR/$1/
W2V_LOC=$INPUT_DIR/$1'_vecs.txt'
SAVE_LOC=$SCRATCH'/output_'$1
SEARCH_TYPE=$2
SEARCH_SPACE=$3
NUM_MODELS=$4
NUM_FOLDS=1
MODEL_FILE=$INPUT_DIR'/'$SEARCH_TYPE'_'$SEARCH_SPACE'_cnn_'$NUM_MODELS'.models'

# pwd
echo $INPUT_DIR
echo $DATASET
echo $W2V_LOC
echo $SAVE_LOC
echo $MODEL_FILE

rm -rf $SAVE_LOC
mkdir -p $SAVE_LOC

for ((i=0; i<$( wc -l $MODEL_FILE | cut -f1 -d' ' ); i++)); do
  python run.py $DATASET $W2V_LOC $SAVE_LOC $NUM_FOLDS -f $MODEL_FILE $i >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt &
done

wait

python eval.py $SAVE_LOC/saved_models/ $DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt

cp -avr $SAVE_LOC $INPUT_DIR/results/
