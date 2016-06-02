
#for DSET in 20_ng_comp 20_ng_religion amazon_reviews trees_binary convote 20_ng_all_topics 20_ng_science
#do
#    for N_MDLS in 1 2
#    do 
#	for MDL_TYP in linear linear-xgboost xgboost
#	do
#	    for N_ITERS in 30 100
#	    do
#
#		qsub -v DATASET=$DSET,NUM_MODELS=$N_MDLS,MODEL_TYPE=$MDL_TYP,NUM_ITERS=$N_ITERS ./one_job.job
#		
#	    done
#	done
#    done
#done


qsub -v DATASET=trees_binary,NUM_MODELS=1,MODEL_TYPE=linear,NUM_ITERS=30,NUM_FOLDS=1 ./one_job.job