
#overwrite protection
if [$1]
  then
    bash train_and_eval.sh ~/datasets/sst2 ~/datasets/output.txt 1 cnn bayesopt 1 ../output/"$1" 1
fi