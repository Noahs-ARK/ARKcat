
import sys
import numpy as np


def extract_single_example(example):
    example[4] = example[4].replace('7e-05', '0.00007')
    example[4] = example[4].replace('5e-05', '0.00005')
    hparams_string = example[4].split(' ')
    hparams = {}
    cur_hparam  = ''
    for hp in hparams_string:
        if hp == '':
            continue
        elif ':' in hp:
            cur_hparam = hp.split(':')[0]
            hparams[cur_hparam] = ''
        else :
            hparams[cur_hparam] += hp
    example[4] = hparams

def extract_examples(lines):
    
    examples = [[]]
    for line in lines:
        line = line.strip()
        if line == "":
            examples.append([])
        else:
            example = line.split(',')
            example[len(example)-2] += "," + example[len(example)-1]
            del example[-1]
            example[0] = float(example[0])
            example[1] = float(example[1])
            example[2] = float(example[2])
            extract_single_example(example)
            examples[len(examples)-1].append(example)

    return [x for x in examples if x != []]

def compute_avg_best_so_far(examples):
    best_so_far = []
    for sample_of_k in examples:
        for i in range(len(sample_of_k)):
            if len(best_so_far) <= i:
                best_so_far.append([])
            best_so_far[i].append(float(sample_of_k[i][2]))
    avg_best = []
    avg_best_ci = []
    for i in range(len(best_so_far)):
        avg_best_ci.append(1.96*np.std(np.asarray(best_so_far[i]))/np.sqrt(len(examples)))
        avg_best.append(np.average(np.asarray(best_so_far[i])))

    if False:
        for i in range(len(best_so_far)):
            print(avg_best[i], avg_best_ci[i])

    return avg_best, avg_best_ci

def frac_set_has_hp_value(examples):
    total_counts = {}
    # this variable counts the number of sets each hparam setting appears in
    num_sets_each_val = {}
    for set_of_k in examples:
        seen_in_cur_set = {}
        for example in set_of_k:
            for hp in example[4]:
                
                # to add to total counts:
                if hp not in total_counts:
                    total_counts[hp] = {}
                v = example[4][hp]
                if v not in total_counts[hp]:
                    total_counts[hp][v] = 0
                total_counts[hp][v] += 1

                # to add to num_sets_each_val:
                if hp not in seen_in_cur_set:
                    seen_in_cur_set[hp] = set()
                seen_in_cur_set[hp].add(v)
        for hp in seen_in_cur_set:
            if hp not in num_sets_each_val:
                num_sets_each_val[hp] = {}
            for val in seen_in_cur_set[hp]:
                if val not in num_sets_each_val[hp]:
                    num_sets_each_val[hp][val] = 0
                num_sets_each_val[hp][val] += 1
    for hparam in num_sets_each_val:
        for val in num_sets_each_val[hparam]:
            num_sets_each_val[hparam][val] = 1.0 * num_sets_each_val[hparam][val] / len(examples)
    for hparam in total_counts:
        for val in total_counts[hparam]:
            total_count = (len(examples) * len(examples[0]))
            total_counts[hparam][val] = 1.0 * total_counts[hparam][val] / total_count

    total_counts_lists = dict_to_ordered_lists(total_counts)
    num_sets_each_val_lists = dict_to_ordered_lists(num_sets_each_val)
    
                
    if False:
        for hparam in total_counts:
            if len(total_counts[thing]) > 2:
                print thing, sorted(total_counts[thing].items())
        for thing in num_sets_each_val:
            if len(num_sets_each_val[thing]) > 2:
                print thing, sorted(num_sets_each_val[thing].items())
    return total_counts_lists, num_sets_each_val_lists


def dict_to_ordered_lists(props):
    # desired shape: 
    # hparam -> 'val' -> [] values
    # hparam -> 'avg' -> [] avgs, ordered by values
    # hparam -> 'prop_sets' -> [] proportion of sets appeared in, ordered by values
    # current shape: 
    # hparam -> vals -> proportions
    hparam_to_list = {}
    for hparam in props:
        if len(props[hparam]) > 1:
            hparam_to_list[hparam] = {}
            hparam_to_list[hparam]['vals'] = sorted(props[hparam].keys())

            hparam_to_list[hparam]['prop'] = [x[1] for x in sorted(props[hparam].items())]
        
    return hparam_to_list
        


def get_avg_acc_by_val(examples):
    avg_acc_by_val = {}
    for set_of_k in examples:
        for example in set_of_k:
            cur_acc = example[0]
            for hp in example[4]:
                if hp not in avg_acc_by_val:
                    avg_acc_by_val[hp] = {}
                cur_val = example[4][hp]
                if cur_val not in avg_acc_by_val[hp]:
                    avg_acc_by_val[hp][cur_val] = []
                avg_acc_by_val[hp][cur_val].append(cur_acc)
    # to make these averages
    for hparam in avg_acc_by_val:
        for val in avg_acc_by_val[hparam]:
            avg_acc_by_val[hparam][val] = np.average(np.asarray(avg_acc_by_val[hparam][val]))
            
    avg_accs = {}
    for hparam in avg_acc_by_val:
        if len(avg_acc_by_val[hparam]) > 1:
            avg_accs[hparam] = [x[1] for x in sorted(avg_acc_by_val[hparam].items())]

    if False:
        for hparam in avg_acc_by_val:
            if len(avg_acc_by_val[hparam]) > 2:
                for val, avg in sorted(avg_acc_by_val[hparam].items()):
                    print hparam, val, avg
    return avg_accs
                
    

def get_avg_and_std_dev(file_loc):
    with open(file_loc) as f:
        lines = f.readlines()
        examples = extract_examples(lines)
        avg_best, avg_best_ci = compute_avg_best_so_far(examples)
        total_counts, num_sets_each_val = frac_set_has_hp_value(examples)
        avg_acc_by_val = get_avg_acc_by_val(examples)
        
        
        
        return avg_best, avg_best_ci, total_counts, num_sets_each_val, avg_acc_by_val


iters = '20'
space = 'reg'
model = 'cnn'

dpp_avg_best, dpp_avg_best_ci, dpp_total_counts, dpp_num_sets_each_val, dpp_avg_acc_by_val = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/output/plotting/results/10_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe=dpp,spce=reg.txt')

rand_avg_best, rand_avg_best_ci, rand_total_counts, rand_num_sets_each_val, rand_avg_acc_by_val = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/output/plotting/results/10_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe=dpp_random,spce=reg.txt')


import sample_histograms
sample_histograms.plot_hists(dpp_total_counts, dpp_avg_acc_by_val)


import scatter_with_error_bars
scatter_with_error_bars.make_scatter(dpp_avg_best, dpp_avg_best_ci, rand_avg_best, rand_avg_best_ci)
