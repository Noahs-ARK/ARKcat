# usage:
# bash azure.sh SEARCH_TYPE RAND_INIT NUM_ITERS SEARCH_SPACE
# example: 
# bash azure.sh spearmint_seq 999 1 arch

SEARCH_TYPE=${1}
RAND_INIT=${2}
NUM_ITERS=${3}
SEARCH_SPACE=${4}

DATA_BASE=/home/jessedd/data/text_cat
DATASET=stanford_sentiment_binary
#DATASET=debugging
DATA_LOC=${DATA_BASE}/${DATASET}
W2V_LOC=${DATA_BASE}/${DATASET}/${DATASET}_vecs.txt
NUM_MODELS=1
MODEL_TYPE=cnn
NUM_FOLDS=3
SAVE_BASE=/home/jessedd/projects/ARKcat/output
    
RUN_INFO=$DATASET,nmodels=$NUM_MODELS,mdl_tpe=$MODEL_TYPE,srch_tpe=$SEARCH_TYPE,spce=$SEARCH_SPACE,iters=$NUM_ITERS
SAVE_LOC=${SAVE_BASE}/${RUN_INFO},rand_init=${RAND_INIT}

# this is a hack to get the dpp on the pythonpath
export PYTHONPATH="${PYTHONPATH}:/home/jessedd/projects/dpp_mixed_mcmc"

START_TIME=$(date +%s)


bash train.sh ${SEARCH_TYPE} ${RAND_INIT} ${NUM_ITERS} ${NUM_FOLDS} ${SEARCH_SPACE} ${SAVE_LOC} ${DATA_LOC} ${W2V_LOC} ${MODEL_TYPE} ${START_TIME}


#bash eval.sh ${SEARCH_TYPE} ${RAND_INIT} ${SAVE_BASE} ${SAVE_LOC} ${DATA_LOC} ${START_TIME}
