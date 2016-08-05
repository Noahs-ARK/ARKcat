INPUT_DIR='/work/04256/tg835555'
DATASET=$INPUT_DIR/$1/
W2V_LOC=$DATASET/$1'_vecs.txt'
SAVE_LOC=$INPUT_DIR'/output_'$1
SEARCH_TYPE=$2
SEARCH_SPACE=$3
NUM_MODELS=$4
MODEL_FILE=$INPUT_DIR'/'$SEARCH_TYPE'_'$SEARCH_SPACE'_cnn_'$NUM_MODELS'.models'

#w2v technically unnecessary for specified wvs

for ((i=0; i<$( wc -l $MODEL_FILE | cut -f1 -d' ' ); i++)); do
  python run.py $DATASET $W2V_LOC $SAVE_LOC 1 -f $MODEL_FILE $i >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt &
done
