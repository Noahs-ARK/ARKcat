# usage:
# bash train_and_eval_spot.sh SEARCH_TYPE RAND_INIT CUR_IP NUM_ITERS SEARCH_SPACE
# example: 
# bash train_and_eval_spot.sh dpp_cos 999 0 10 reg_bad_lr

SEARCH_TYPE=${1}
RAND_INIT=${2}
CUR_IP=${3}
NUM_ITERS=${4}
SEARCH_SPACE=${5}

DATA_BASE=/home/ec2-user/data/text_cat
#DATA_BASE=/homes/gws/jessedd/data/text_cat # this is for running jobs on pinot
DATASET=stanford_sentiment_binary
#DATASET=debugging
DATA_LOC=${DATA_BASE}/${DATASET}
W2V_LOC=${DATA_BASE}/${DATASET}/${DATASET}_vecs.txt
NUM_MODELS=1
MODEL_TYPE=cnn
NUM_FOLDS=5
SAVE_BASE=/home/ec2-user/projects/ARKcat/output
#SAVE_BASE=/homes/gws/jessedd/projects/ARKcat/output # this is for running jobs on pinot
    
RUN_INFO=$DATASET,nmodels=$NUM_MODELS,mdl_tpe=$MODEL_TYPE,srch_tpe=$SEARCH_TYPE,spce=$SEARCH_SPACE,iters=$NUM_ITERS
SAVE_LOC=${SAVE_BASE}/${RUN_INFO},rand_init=${RAND_INIT}

# this is a hack to get the dpp on the pythonpath
export PYTHONPATH="${PYTHONPATH}:/home/ec2-user/projects/dpp_mixed_mcmc"



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
ARCHIVE_DIR=${SAVE_BASE}/archive/${SEARCH_TYPE}_${RAND_INIT}_$(date +%s)/
mkdir -p $ARCHIVE_DIR
cp -ar $SAVE_LOC $ARCHIVE_DIR



######### this is to copy from one ec2 instance to another ###########                                                                                                    

KEYPAIR_LOC=/home/ec2-user/projects/ARKcat/aws/jesse-key-pair-uswest2.pem
EC2_STORAGE_DIR=/home/ec2-user/projects/ARKcat/output/archive_${RUN_INFO}/${SEARCH_TYPE}_${RAND_INIT}_$(date +%s)
ssh -i ${KEYPAIR_LOC} -oStrictHostKeyChecking=no ec2-user@${CUR_IP} "mkdir -p $EC2_STORAGE_DIR"
scp -i ${KEYPAIR_LOC} -oStrictHostKeyChecking=no $SAVE_LOC/outfile.txt ec2-user@${CUR_IP}:$EC2_STORAGE_DIR
scp -i ${KEYPAIR_LOC} -oStrictHostKeyChecking=no $SAVE_LOC/errfile.txt ec2-user@${CUR_IP}:$EC2_STORAGE_DIR

#to kill the current instance
#aws ec2 terminate-instances --instance-ids $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
#ssh -i ~/jesse-key-pair-uswest2.pem -oStrictHostKeyChecking=no ec2-user@${CUR_IP} "aws ec2 terminate-instances --instance-ids ${SPOT_INST_ID}"
