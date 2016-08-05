DATASET='~/datasets/sst2'
W2V_LOC='~/datasets/sst2_vecs.txt'
SAVE_LOC='../output/foo_bar'
SEARCH_TYPE='rand'
SEARCH_SPACE='reg'
NUM_MODELS='4'
MODEL_FILE=~/datasets/$SEARCH_TYPE'_'$SEARCH_SPACE'_cnn_'$NUM_MODELS'.models'

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
for ((i=0; i<$NUM_MODELS; i++)); do
  python run.py $DATASET $W2V_LOC $SAVE_LOC 1 -f $MODEL_FILE $i >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt &
done

echo "done with run.py. now going to eval.py"
python eval.py $SAVE_LOC/saved_models/ $DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"
