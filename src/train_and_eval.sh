#DATASET=trees_binary
#DATASET=convote
#DATASET=amazon_reviews
DATASET=$1
NUM_MODELS=$2
MODEL_TYPE=$3
NUM_ITERS=$4
SAVE_LOC=$5


rm -rf $SAVE_LOC
echo "making the dir to save output..."
mkdir -p $SAVE_LOC
echo $1
echo $2
echo $3
echo $4
echo `pwd`

echo "about to run.py"
python run.py /cab1/corpora/bayes_opt/$DATASET/ $SAVE_LOC $NUM_MODELS $MODEL_TYPE -m $NUM_ITERS > $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
echo "done with run.py. now going to eval.py"
python eval.py $SAVE_LOC/saved_models/ /cab1/corpora/bayes_opt/$DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"

#because i'm running out of space lol
rm -r $SAVE_LOC/saved_models/