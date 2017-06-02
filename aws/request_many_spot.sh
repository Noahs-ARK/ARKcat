


if [ "$STY" = "" ]
then
    echo "not in screen! please start a screen session to run this script"
    exit 1
fi



# options: dpp_ham, dpp, dpp_random, random, bayes_opt
# options: reg, reg_half_lr, reg_bad_lr, arch



for SRCH_TPE in bayes_opt random; do
    for i in `seq 0 9`; do 
	for j in `seq $((i * 10)) $(( $((i * 10)) + 9)) `; do
	    bash request_spot.sh ${SRCH_TPE} ${j} 20 reg_half_lr &
	    #echo ${SRCH_TPE} ${j} 20 reg_half_lr &
	    sleep 2
	done
	sleep 200
    done
done

sleep 28800

for SRCH_TPE in bayes_opt random; do
    for i in `seq 0 9`; do 
	for j in `seq $((i * 10)) $(( $((i * 10)) + 9)) `; do
	    bash request_spot.sh ${SRCH_TPE} ${j} 20 reg &
	    #echo ${SRCH_TPE} ${j} 20 reg &
	    sleep 2
	done
	sleep 200
    done
done
