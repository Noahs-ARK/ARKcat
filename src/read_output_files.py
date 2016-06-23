import os, sys

#to read the results:


output_dir = sys.argv[1]


def print_table_header(num_iters):
    header = """\\begin{table}[h]
\\centering
\\caption{
My accuracy vs Yogatama et al 2015 accuracy (""" + num_iters + """ iterations)
\\label{tbl:test_acc}
}
\\small \\begin{tabular}{|@{\\hspace{1.0mm}}c@{\\hspace{1.0mm}}|@{\\hspace{1.0mm}}c@{\\hspace{1.0mm}}|r|r|r|r|r|r|r|r|}
\\hline
\\abovespace
& \\textbf{Dataset} & Dani acc & LR & XGBoost & LR or XGB & LR and LR &  XGB and XGB & LR and XGB
\\belowspace
\\\\
\\hline
\\abovespace
\\multirow{4}{*}{\\rotatebox{90}{\\bf Other}}
"""
    print(header)

dataset_names = {'trees_binary': 'Stanford sentiment', 'amazon_reviews':'Amazon electronics', 'imdb_reviews':'IMDB reviews','convote':'Congress vote', '20_ng_all': '20N all topics', 
                 '20_ng_science':'20N all science', '20_ng_religion':'20N athiest.religion', '20_ng_comp':'20N x.graphics'}
dani_acc = {'trees_binary': '82.43', 'amazon_reviews':'91.56', 'imdb_reviews':'90.85','convote':'78.59', '20_ng_all': '87.84', 
                 '20_ng_science':'95.82', '20_ng_religion':'86.32', '20_ng_comp':'92.09'}

def print_table_mid():

    mid = """

\\belowspace
\\\\
\\hline \\hline
\\abovespace
\\multirow{4}{*}{\\rotatebox{90}{\\bf 20N}}

"""
    sys.stdout.write(mid)

def print_table_end():
    print('\\\\')
    print('')
    print('\\belowspace')
    print('\\\\')
    print('\\hline')
    print('\\end{tabular}')
    print('\\end{table}')



for N_ITERS in ["100", "30"]:
    print('\n\n\n')
    print_table_header(N_ITERS)
    for DSET in ['trees_binary', 'amazon_reviews', 'imdb_reviews','convote', '20_ng_all', '20_ng_science', '20_ng_religion', '20_ng_comp']:
        
        sys.stdout.write(' & ' + dataset_names[DSET] + ' & ' + dani_acc[DSET])
        for N_MDLS in ["1", "2"]:
            for MDL_TYP in ["linear", "xgboost", "linear-xgboost"]:
                
                #amazon_reviews,nmodels=1,type=linear,iters=100
                sub_dir = DSET + ',nmodels=' + N_MDLS + ',type=' + MDL_TYP + ',iters=' + N_ITERS + '/'
                if not os.path.isfile(output_dir + sub_dir + 'outfile.txt'):
                    sys.stdout.write(' & 0')
                else:
                    lines = open(output_dir + sub_dir + 'outfile.txt').readlines()
                    found_best_models = False
                    printed_best_test = False
                    printed_best_dev = False
                    for line in lines:
                        if line == "best models:\n":
                            found_best_models = True
                            sys.stdout.write(' & ')
                        elif found_best_models and not printed_best_test:
                            sys.stdout.write(str(round(10000*float(line.split(':')[1].strip()))/100))
                            printed_best_test = True
                        elif found_best_models and not printed_best_dev:
                            sys.stdout.write(' (' + str(round(10000*float(line.split(':')[1].strip()))/100) + ')')
                            printed_best_dev = True
        if DSET == 'convote':
            print_table_mid()
        elif DSET == '20_ng_comp':
            print_table_end()
        else:
            sys.stdout.write(' \\\\\n')


print('\n\n\n')
    
#a = 
""" & Stanford sentiment &  82.43 & 80.73 (80.51) & 70.76 (69.08) & 80.73 (80.34) & 80.16 (81.66) & 71.33 (68.15) & 80.62 (80.56) \\
 & Amazon electronics & 91.56 & 0 & 0 & 0& 0 & 0 & 0\\
 & IMDB reviews & 90.85 & 0 & 0 & 0 & 0 & 0 & 0\\
 & congress vote & 78.59 & 77.88 (68.86) & 78.76 (74.94) &  &  &  &    \belowspace
\\
\hline \hline
\abovespace
\multirow{4}{*}{\rotatebox{90}{\bf 20N}}
& all &  87.84 & 0 & 0 & 0 & 0 & 0 & 0\\
& science & 95.82 &0 & 0 & 0 & 0 & 0 & 0\\
& religion & 86.32 & 91.23 (72.63) & 80.12 (63.16) & 90.64 (70.0) & 90.64 (73.16) & 78.95 (62.81) & 90.06 (71.93) \\
& x.graphics &  92.09 & 86.38 (78.95) & 80.85 (73.85) & 84.68 (75.38) & 86.38 (78.06) & 80.85 (72.19) & 85.53 (78.06) \\\belowspace
\\
"""
