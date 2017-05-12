
for SRCH_TPE in dpp_ham; do
    for i in `seq 0 9`; do 
	for j in `seq $((i * 10)) $(( $((i * 10)) + 9)) `; do
	    bash request_spot.sh ${SRCH_TPE} ${j} 20 reg_half_lr &
	    #echo ${SRCH_TPE} ${j}_lr=10to3
	    sleep 2
	done
	sleep 200
    done
done


# example:
# bash request_spot.sh dpp_ham 0 10 &
