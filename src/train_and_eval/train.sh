SEARCH_TYPE=${1}
RAND_INIT=${2}
NUM_ITERS=${3}
BATCH_SIZE=${4}
NUM_FOLDS=${5}
SEARCH_SPACE=${6}
SAVE_LOC=${7}
DATA_LOC=${8}
W2V_LOC=${9}
MODEL_TYPE=${10}
START_TIME=${11}


cd ..

if [ -d "$SAVE_LOC" ]; then
    echo "removing current save location: $SAVE_LOC"
    rm -rf $SAVE_LOC
fi
echo "making the dir to save output..."
mkdir -p $SAVE_LOC/


echo "about to run.py"
python controller.py  $DATA_LOC/ $W2V_LOC $SAVE_LOC $NUM_FOLDS $SEARCH_TYPE -b $MODEL_TYPE $SEARCH_SPACE $NUM_ITERS ${BATCH_SIZE} #> $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
RUN_TIME=$(date +%s)
echo "done with run.py."
echo 'run time:'
echo $(($RUN_TIME - $START_TIME))
