DATASET=$1
W2V_LOC=$2
SAVE_LOC=$3
SEARCH_TYPE=$4
SEARCH_SPACE=$5
NUM_FOLDS=$6
LINE_NUM=$7
NUM_MODELS=$8
MODEL_FILE=~/datasets/$SEARCH_TYPE'_'$SEARCH_SPACE'_cnn_'$NUM_MODELS'.models'

echo $SAVE_LOC
rm -rf $SAVE_LOC
echo "making the dir to save output..."
mkdir -p $SAVE_LOC /

echo "about to run.py"
echo 'dataset'  $DATASET
echo 'word'$W2V_LOC
echo 'save'$SAVE_LOC
echo 'folds'$NUM_FOLDS
echo 'file'$MODEL_FILE
echo 'line'$LINE_NUM
python run.py $DATASET $W2V_LOC $SAVE_LOC $NUM_FOLDS -f $MODEL_FILE $LINE_NUM >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with run.py. now going to eval.py"
python eval.py $SAVE_LOC/saved_models/ $DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"
