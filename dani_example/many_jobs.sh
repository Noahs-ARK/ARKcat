#qsub -v DATASET=/cab1/corpora/bayes_opt/trees_binary/,PENALTY=l2,TOL=0.098,STR=10,NMIN=1,NMAX=2,BINARY=True,IDF=True,STOP=False ./one_job.job
#qsub -v DATASET=/cab1/corpora/bayes_opt/amazon_reviews/dani_format/,PENALTY=l2,TOL=0.022,STR=120,NMIN=1,NMAX=3,BINARY=True,IDF=True,STOP=False ./one_job.job



for i in `seq -5 5`;
do
    for j in `seq -5 5`;
    do

	qsub -v DATASET=/cab1/corpora/bayes_opt/trees_binary/,PENALTY=l2,TOL=`echo 0.098 + 0.0001*$j | bc`,STR=`echo 10 + 0.1*$i | bc`,NMIN=1,NMAX=2,BINARY=False,IDF=True,STOP=False ./one_job.job
	#qsub -v DATASET=/cab1/corpora/bayes_opt/amazon_reviews/dani_format/,PENALTY=l2,TOL=`echo 0.022 + 0.0001*$j | bc`,STR=`echo 120 + 0.1*$i | bc`,NMIN=1,NMAX=3,BINARY=True,IDF=True,STOP=False ./one_job.job
	#qsub -v DATASET=/cab1/corpora/bayes_opt/convote/,PENALTY=l2,TOL=0.012$i,STR=121.$j,NMIN=2,NMAX=2,BINARY=True,IDF=$k,STOP=False ./one_job.job

    done
done