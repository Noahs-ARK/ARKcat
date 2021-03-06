# possible good search spaces: reg, reg_bad_lr, reg_half_lr, reg_good_lr, arch, dropl2learn_bad_lr, dropl2learn_half_lr, dropl2learn_good_lr

SEARCH_TYPE=${1}
RAND_INIT=${2}
NUM_ITERS=${3}
NUM_FOLDS=${4}
SEARCH_SPACE=${5}
SAVE_LOC=${6}
DATA_LOC=${7}
W2V_LOC=${8}
MODEL_TYPE=${9}
START_TIME=${10}


cd ..

if [ -d "$SAVE_LOC" ]; then
    echo "removing current save location: $SAVE_LOC"
    rm -rf $SAVE_LOC
fi
echo "making the dir to save output..."
mkdir -p $SAVE_LOC/


echo "about to run.py"
python controller.py  $DATA_LOC/ $W2V_LOC $SAVE_LOC $NUM_FOLDS $SEARCH_TYPE -b $MODEL_TYPE $SEARCH_SPACE $NUM_ITERS > $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
RUN_TIME=$(date +%s)
echo "done with run.py."
echo 'run time:'
echo $(($RUN_TIME - $START_TIME))
