import os, sys

#to read the results:


output_dir = "/home/jessed/ARKcat/output/"


for N_ITERS in ["100", "30"]:
    for DSET in ["trees_binary", "amazon_reviews",  "convote",  "20_ng_religion", "20_ng_comp"]:

        for N_MDLS in ["1", "2"]:
            for MDL_TYP in ["linear", "xgboost", "linear-xgboost"]:

                #amazon_reviews,nmodels=1,type=linear,iters=100
                sub_dir = DSET + ',nmodels=' + N_MDLS + ',type=' + MDL_TYP + ',iters=' + N_ITERS + '/'
                if not os.path.isfile(output_dir + sub_dir + 'outfile.txt'):
                    print sub_dir
                else:
                    lines = open(output_dir + sub_dir + 'outfile.txt').readlines()
                    found_best_models = False
                    printed_best_test = False
                    printed_best_dev = False
                    for line in lines:
                        if line == "best models:\n":
                            found_best_models = True
                        elif found_best_models and not printed_best_test:
                            sys.stdout.write(str(round(10000*float(line.split(':')[1].strip()))/100))
                            printed_best_test = True
                        elif found_best_models and not printed_best_dev:
                            sys.stdout.write(' (' + str(round(10000*float(line.split(':')[1].strip()))/100) + ')')
                            printed_best_dev = True
                    if N_MDLS == "2" and MDL_TYP == "linear-xgboost":
                        sys.stdout.write(' \\\\ ' + DSET + ' , iters=' + N_ITERS + '\n')
                    else:
                        sys.stdout.write(' & ')
                        
