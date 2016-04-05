#DATASET=trees_binary
DATASET=convote

rm -r ../output
python run.py ../../data/$DATASET/train.json ../../data/$DATASET/train.csv ../../data/$DATASET/dev.json ../../data/$DATASET/dev.csv ../output
python eval.py ../output/saved_models/ ../../data/$DATASET/ ../output/

#python run.py ../example_data/train.json ../example_data/train.csv ../example_data/dev.json ../example_data/dev.csv ../output

#python run.py ../../data/atheistchristian/train.json ../../data/atheistchristian/train.csv ../../data/atheistchristian/dev.json ../../data/atheistchristian/dev.csv ../output

#python run.py ../../data/trees_binary/train2.json ../../data/trees_binary/train2.csv ../../data/trees_binary/dev_train.json ../../data/trees_binary/dev_train.csv ../output


#python run.py ../../data/convote/train.json ../../data/convote/train.csv ../../data/convote/dev.json ../../data/convote/dev.csv ../output

#python run.py ../../data/amazon_reviews/train.json ../../data/amazon_reviews/train.csv ../../data/amazon_reviews/dev.json ../../data/amazon_reviews/dev.csv ../output
