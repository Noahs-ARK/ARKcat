

#DATASET=trees_binary
#DATASET=convote
#DATASET=amazon_reviews
DATASET=$1
W2V_LOC=$2
NUM_MODELS=$3
MODEL_TYPE=$4
SEARCH_TYPE=$5
SEARCH_SPACE=$6
NUM_ITERS=$7
SAVE_LOC=$8
NUM_FOLDS=$9

echo $SAVE_LOC
#prevent accidental overwrites
if [$SAVE_LOC != '../output/']
  then
    rm -rf $SAVE_LOC
fi
echo "making the dir to save output..."
mkdir -p $SAVE_LOC /

echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo `pwd`

echo "about to run.py"
#python run.py ~/datasets/sst2/  ~/datasets/output.txt ../output/dummy 1 cnn bayesopt -m 2 1 > ../output/dummy/outfile.txt 2> ../output/dummy/errfile.txt
python run.py $DATASET/ $W2V_LOC $SAVE_LOC $NUM_MODELS $MODEL_TYPE $SEARCH_TYPE $SEARCH_SPACE -m $NUM_ITERS $NUM_FOLDS > $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
echo "done with run.py. now going to eval.py"
python eval.py $SAVE_LOC/saved_models/ $DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"
