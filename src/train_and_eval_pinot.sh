DATA_BASE=/home/ec2-user/data/text_cat
#DATA_BASE=/homes/gws/jessedd/data/text_cat # this is for running jobs on pinot
DATASET=stanford_sentiment_binary
#DATASET=debugging
DATA_LOC=${DATA_BASE}/${DATASET}
W2V_LOC=${DATA_BASE}/${DATASET}/${DATASET}_vecs.txt
NUM_MODELS=1
MODEL_TYPE=cnn
SEARCH_TYPE=${1}
SEARCH_SPACE=reg
NUM_ITERS=3
NUM_FOLDS=5
SAVE_BASE=/home/ec2-user/projects/ARKcat/output
#SAVE_BASE=/homes/gws/jessedd/projects/ARKcat/output # this is for running jobs on pinot

    

SAVE_LOC=${SAVE_BASE}/$DATASET,nmodels=$NUM_MODELS,mdl_tpe=$MODEL_TYPE,srch_tpe=$SEARCH_TYPE,spce=$SEARCH_SPACE,iters=$NUM_ITERS,rand_init=${2}



if [ -d "$SAVE_LOC" ]; then
    echo "removing current save location: $SAVE_LOC"
    rm -rf $SAVE_LOC
fi
echo "making the dir to save output..."
mkdir -p $SAVE_LOC/



echo "about to run.py"
START_TIME=$(date +%s)
python run.py  $DATA_LOC/ $W2V_LOC $SAVE_LOC $NUM_FOLDS $SEARCH_TYPE -b $MODEL_TYPE $SEARCH_SPACE $NUM_ITERS > $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
RUN_TIME=$(date +%s)
echo "done with run.py."
echo 'run time:'
echo $(($RUN_TIME - $START_TIME))


python eval.py $SAVE_LOC/saved_models/ $DATA_LOC/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"
echo $8
echo 'eval time:'
echo $(($(date +%s) - $RUN_TIME))
ARCHIVE_DIR=${SAVE_BASE}/archive/${1}_${2}_$(date +%s)/
mkdir -p $ARCHIVE_DIR
cp -ar $SAVE_LOC $ARCHIVE_DIR



######### this is to copy from one ec2 instance to another ###########                                                                                                    

EC2_STORAGE_DIR=/home/ec2-user/projects/ARKcat/output/archive/${1}_${2}_$(date +%s)
ssh -i ~/jesse-key-pair-uswest2.pem -oStrictHostKeyChecking=no ec2-user@${3} "mkdir -p $EC2_STORAGE_DIR"
scp -i "/home/ec2-user/jesse-key-pair-uswest2.pem" -oStrictHostKeyChecking=no -r $SAVE_LOC ec2-user@${3}:$EC2_STORAGE_DIR
ssh -i ~/jesse-key-pair-uswest2.pem -oStrictHostKeyChecking=no ec2-user@${3} "aws ec2 terminate-instances --instance-ids ${4}"
