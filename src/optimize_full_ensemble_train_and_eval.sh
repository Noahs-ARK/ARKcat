DATASET=trees_binary

rm -r ../output
python optimize_full_ensemble.py ../../data/$DATASET/train.json ../../data/$DATASET/train.csv ../../data/$DATASET/dev.json ../../data/$DATASET/dev.csv ../output
#python eval.py ../output/saved_models/ ../../data/$DATASET/ ../output/
