DATASET=$1
W2V_LOC=$2
SAVE_LOC=$3
SEARCH_TYPE=$4
SEARCH_SPACE=$5
NUM_MODELS=$6
NUM_PROCS=$7
MODEL_FILE=~/datasets/$SEARCH_TYPE'_'$SEARCH_SPACE'_cnn_'$NUM_MODELS'.models'
curr_procs=0

echo $1
echo $2
echo "Number of processes:" $7

echo $MODEL_FILE
rm $MODEL_FILE
echo "creating models..."

python print_grid_or_rand_search.py $SEARCH_SPACE '--'$SEARCH_TYPE $NUM_MODELS

echo $SAVE_LOC
rm -rf $SAVE_LOC
echo "making the dir to save output..."
mkdir -p $SAVE_LOC /

echo "about to run.py"
for i in $(seq 0 $( wc -l $MODEL_FILE | cut -f1 -d' ' )); do
  python run.py $DATASET $W2V_LOC $SAVE_LOC 1 -f $MODEL_FILE $i >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt & (( ++curr_procs ))
  while (( curr_procs >= NUM_PROCS )); do
    wait && (( NUM_PROCS-- ))
  done
done

echo "done with run.py. now going to eval.py"
python eval.py $SAVE_LOC/saved_models/ $DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"
