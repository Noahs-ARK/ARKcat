# usage:
# bash azure.sh SEARCH_TYPE RAND_INIT CUR_IP NUM_ITERS BATCH_SIZE SEARCH_SPACE
# example: 
# bash azure.sh spearmint_seq 999 0 1 3 arch

SEARCH_TYPE=${1}
RAND_INIT=${2}
CUR_IP=${3}
NUM_ITERS=${4}
BATCH_SIZE=${5}
SEARCH_SPACE=${6}


DATA_BASE=/home/jessedd/data/text_cat
DATASET=stanford_sentiment_binary
#DATASET=debugging
DATA_LOC=${DATA_BASE}/${DATASET}
W2V_LOC=${DATA_BASE}/${DATASET}/${DATASET}_vecs.txt
NUM_MODELS=1
MODEL_TYPE=cnn
NUM_FOLDS=3
SAVE_BASE=/home/jessedd/projects/ARKcat/output
    
RUN_INFO=$DATASET,nmodels=$NUM_MODELS,mdl_tpe=$MODEL_TYPE,srch_tpe=$SEARCH_TYPE,spce=$SEARCH_SPACE,iters=$NUM_ITERS,batchsze=${BATCH_SIZE}
SAVE_LOC=${SAVE_BASE}/${RUN_INFO},rand_init=${RAND_INIT}

# this is a hack to get the dpp and spearmint on the pythonpath
export PYTHONPATH="${PYTHONPATH}:/home/jessedd/projects/dpp_mixed_mcmc:/home/jessedd/projects/spearmint/spearmint-lite:/home/jessedd/projects/spearmint/spearmint"

START_TIME=$(date +%s)


bash train.sh ${SEARCH_TYPE} ${RAND_INIT} ${NUM_ITERS} ${BATCH_SIZE} ${NUM_FOLDS} ${SEARCH_SPACE} ${SAVE_LOC} ${DATA_LOC} ${W2V_LOC} ${MODEL_TYPE} ${START_TIME}

exit

bash eval.sh ${SEARCH_TYPE} ${RAND_INIT} ${SAVE_BASE} ${SAVE_LOC} ${DATA_LOC} ${START_TIME}


KEYPAIR_LOC=/home/jessedd/jesse-key-pair-uswest2.pem
AZURE_STORAGE_DIR=/home/jessedd/projects/ARKcat/output/archive_${RUN_INFO}/${SEARCH_TYPE}_${RAND_INIT}_$(date +%s)
ssh -i ${KEYPAIR_LOC} -oStrictHostKeyChecking=no jessedd@${CUR_IP} "mkdir -p ${AZURE_STORAGE_DIR}"
scp -i ${KEYPAIR_LOC} -oStrictHostKeyChecking=no $SAVE_LOC/outfile.txt jessedd@${CUR_IP}:$AZURE_STORAGE_DIR
scp -i ${KEYPAIR_LOC} -oStrictHostKeyChecking=no $SAVE_LOC/errfile.txt jessedd@${CUR_IP}:$AZURE_STORAGE_DIR
