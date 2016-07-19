
#incorporate word vector file location??

#DATASET=trees_binary
#DATASET=convote
#DATASET=amazon_reviews
DATASET=$1
W2V_LOC=$2
NUM_MODELS=$3
MODEL_TYPE=$4
NUM_ITERS=$5
SAVE_LOC=$6
NUM_FOLDS=$7

echo $SAVE_LOC

rm -rf $SAVE_LOC
echo "making the dir to save output..."
mkdir -p $SAVE_LOC /

echo $1
echo $2
echo $3
echo $4
echo $5
echo `pwd`

echo "about to run.py"

python run.py $DATASET/ $W2V_LOC $SAVE_LOC $NUM_MODELS $MODEL_TYPE -m $NUM_ITERS $NUM_FOLDS > $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
echo "done with run.py. now going to eval.py"
python eval.py $SAVE_LOC/saved_models/ $DATASET/ $W2V_LOC $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"
