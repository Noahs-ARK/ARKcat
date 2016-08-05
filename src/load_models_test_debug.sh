# $out="./tmpo.txt"
# $err="./tmpe.txt"

python run.py /home/katya/datasets/sst2/ /home/katya/datasets/sst2_vecs.txt ../output/foo_bar 5 -f /home/katya/datasets/rand_reg_cnn_4.models 3 >> "tmpo.txt" 2>> "tmpe.txt" &
python run.py /home/katya/datasets/sst2/ /home/katya/datasets/sst2_vecs.txt ../output/foo_bar 5 -f /home/katya/datasets/rand_reg_cnn_4.models 2 >> "tmpo.txt" 2>> "tmpe.txt" &
