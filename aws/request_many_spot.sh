

for i in `seq 2 6`; do 
    for j in `seq $((i * 10)) $(( $((i * 10)) + 9)) `; do
	bash request_spot.sh dpp_random $j 50 &
	#echo $j
    done
    sleep 2
done
