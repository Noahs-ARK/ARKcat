#overwrite protection
if [ -z "$1" ]; then
    echo "No argument supplied"
else
  bash train_and_eval.sh ~/datasets/sst2 ~/datasets/output.txt 1 cnn grid_search reg 5 ../output/"$1" 1
fi
