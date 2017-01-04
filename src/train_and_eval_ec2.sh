DATA_BASE=/home/ec2-user/datasets
DATASET=stanford_sentiment_binary
#DATASET=debugging
DATA_LOC=${DATA_BASE}/${DATASET}
W2V_LOC=${DATA_BASE}/${DATASET}/${DATASET}_vecs.txt
NUM_MODELS=1
MODEL_TYPE=cnn
SEARCH_TYPE=bayes_opt
SEARCH_SPACE=reg
NUM_ITERS=7
NUM_FOLDS=1
SAVE_LOC=/data/output/$DATASET,nmodels=$NUM_MODELS,mdl_tpe=$MODEL_TYPE,srch_tpe=$SEARCH_TYPE,spce=$SEARCH_SPACE,iters=$NUM_ITERS


echo "save location: $SAVE_LOC"
rm -rf $SAVE_LOC
echo "making the dir to save output..."
mkdir -p $SAVE_LOC/

echo 'start time:'
START_TIME=$(date +%s)
echo $START_TIME

echo "about to run.py"
python run.py  $DATA_LOC/ $W2V_LOC $SAVE_LOC $NUM_FOLDS  -b $MODEL_TYPE $SEARCH_SPACE $NUM_ITERS > $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
echo "done with run.py."
RUN_TIME=$(date +%s)
echo 'run time:'
echo $RUN_TIME
echo $(($RUN_TIME - $START_TIME))


#python eval.py $SAVE_LOC/saved_models/ $DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
#echo "done with eval.py"
#echo $8
#echo 'eval time:'
#echo $(date +%s)
#echo $(($(date +%s) - $RUN_TIME))
#cp -avr $SAVE_LOC ../../archive/$(date +%s)/
