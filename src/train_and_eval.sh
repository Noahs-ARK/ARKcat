#DATASET=trees_binary
#DATASET=convote
#DATASET=amazon_reviews
DATASET=$1
NUM_MODELS=$2
MODEL_TYPE=$3
NUM_ITERS=$4
SAVE_LOC=$5

rm -r $SAVE_LOC
mkdir -p $SAVE_LOC
echo $1
echo $2
echo $3
echo $4
echo `pwd`

python run.py ../../data/$DATASET/train.json ../../data/$DATASET/train.csv ../../data/$DATASET/dev.json ../../data/$DATASET/dev.csv $SAVE_LOC $NUM_MODELS $MODEL_TYPE -m $NUM_ITERS > $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
python eval.py $SAVE_LOC/saved_models/ ../../data/$DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt