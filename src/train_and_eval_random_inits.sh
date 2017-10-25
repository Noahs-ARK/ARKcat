NUM_RANDOM_INITS=50

#for i in `seq 31 $NUM_RANDOM_INITS`; do
for i in `seq 1 25`; do
    bash train_and_eval_spot.sh dpp $i &

done
