


if [ "$STY" = "" ]
then
    echo "not in screen! please start a screen session to run this script"
    exit 1
fi



# options: dpp_ham, dpp_cos, dpp_l2, dpp_random, random, bayes_opt
# options: reg, reg_half_lr, reg_bad_lr, arch


for SPACE in reg_bad_lr; do
    for SRCH_TPE in dpp_l2; do
	for i in `seq 0 9`; do 
	    for j in `seq $((i * 10)) $(( $((i * 10)) + 9)) `; do
		bash request_spot.sh ${SRCH_TPE} 0${j} 20 ${SPACE} &
		#echo ${SRCH_TPE} ${j} 20 reg_half_lr &
		sleep 2
	    done
	    sleep 200
	done
    done
    sleep 2000
done
