SEARCH_TYPE=${1}
RAND_INIT=${2}
SAVE_BASE=${3}
SAVE_LOC=${4}
DATA_LOC=${5}
START_TIME=${6}


cd ..

EVAL_START_TIME=$(date +%s)
python eval.py $SAVE_LOC/saved_models/ $DATA_LOC/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"
echo 'eval time:'
echo $(($(date +%s) - $EVAL_START_TIME))
echo "total train and eval time: $(($(date +%s) - $START_TIME))" >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
ARCHIVE_DIR=${SAVE_BASE}/archive/${SEARCH_TYPE}_${RAND_INIT}_$(date +%s)/
mkdir -p $ARCHIVE_DIR
cp -ar $SAVE_LOC $ARCHIVE_DIR

