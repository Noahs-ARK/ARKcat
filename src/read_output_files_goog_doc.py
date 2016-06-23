import os, sys

#to read the results:


output_dir = sys.arg[1]

dataset_names = {'trees_binary': 'Stanford sentiment', 'amazon_reviews':'Amazon electronics', 'imdb_reviews':'IMDB reviews','convote':'Congress vote', '20_ng_all': '20N all topics', 
                 '20_ng_science':'20N all science', '20_ng_religion':'20N athiest.religion', '20_ng_comp':'20N x.graphics'}
dani_acc = {'trees_binary': '82.43', 'amazon_reviews':'91.56', 'imdb_reviews':'90.85','convote':'78.59', '20_ng_all': '87.84', 
                 '20_ng_science':'95.82', '20_ng_religion':'86.32', '20_ng_comp':'92.09'}


for N_ITERS in ["100", "30"]:
    for DSET in ['trees_binary', 'amazon_reviews', 'imdb_reviews','convote', '20_ng_all', '20_ng_science', '20_ng_religion', '20_ng_comp']:
        
        sys.stdout.write('  ' + dataset_names[DSET] + ' , ' + dani_acc[DSET])
        for N_MDLS in ["1", "2"]:
            for MDL_TYP in ["linear", "xgboost", "linear-xgboost"]:
                
                #amazon_reviews,nmodels=1,type=linear,iters=100
                sub_dir = DSET + ',nmodels=' + N_MDLS + ',type=' + MDL_TYP + ',iters=' + N_ITERS + '/'
                if not os.path.isfile(output_dir + sub_dir + 'outfile.txt'):
                    sys.stdout.write(' , 0')
                else:
                    lines = open(output_dir + sub_dir + 'outfile.txt').readlines()
                    found_best_models = False
                    printed_best_test = False
                    printed_best_dev = False
                    for line in lines:
                        if line == "best models:\n":
                            found_best_models = True
                            sys.stdout.write(' , ')
                        elif found_best_models and not printed_best_test:
                            sys.stdout.write(str(round(10000*float(line.split(':')[1].strip()))/100))
                            printed_best_test = True
                        elif found_best_models and not printed_best_dev:
                            sys.stdout.write(' (' + str(round(10000*float(line.split(':')[1].strip()))/100) + ')')
                            printed_best_dev = True
        sys.stdout.write(' \n')
    sys.stdout.write(' \n')


