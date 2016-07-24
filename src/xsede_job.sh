#overwrite protection
if [ -z "$3" ]; then
    echo "No argument supplied"
else
  bash train_and_eval.sh $WORK/sst2 $WORK/output.txt 1 $1 $2 $3 ../output/"$4" 1
fi
