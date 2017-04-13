NUM_RANDOM_INITS=50

for i in `seq 31 $NUM_RANDOM_INITS`; do
    bash train_and_eval_pinot.sh dpp_random $i 

done
