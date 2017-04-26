
for i in `seq 1 10`; do
    bash request_spot.sh dpp $i &
done
