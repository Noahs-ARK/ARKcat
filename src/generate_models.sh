echo "about to run.py"
#python run.py ~/datasets/sst2/  ~/datasets/output.txt ../output/dummy 1 cnn bayesopt -m 2 1 > ../output/dummy/outfile.txt 2> ../output/dummy/errfile.txt
python run.py ~/datasets/sst2/  ~/datasets/output.txt ../output/dummy - -m $NUM_ITERS $NUM_FOLDS > $SAVE_LOC/outfile.txt 2> $SAVE_LOC/errfile.txt
echo "done with run.py. now going to eval.py"
echo $8
RUN_TIME=$(date +%s)
echo 'run time:'
echo $RUN_TIME
echo $(($RUN_TIME - $START_TIME))
python eval.py $SAVE_LOC/saved_models/ $DATASET/ $SAVE_LOC/ >> $SAVE_LOC/outfile.txt 2>> $SAVE_LOC/errfile.txt
echo "done with eval.py"
echo $8
echo 'eval time:'
echo $(date +%s)
echo $(($(date +%s) - $RUN_TIME))
cp -avr $SAVE_LOC ../../archive/$(date +%s)/
