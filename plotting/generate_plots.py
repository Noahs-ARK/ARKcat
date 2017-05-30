import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sample_histograms
import scatter_with_error_bars

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
    for hparam in hparams:
        if 'learning_rate' in hparam:
            hparams[hparam] = float(hparams[hparam])
        elif 'reg_strength' in hparam:
            hparams[hparam] = round(np.exp(float(hparams[hparam])),5)
        elif 'dropout' in hparam:
            hparams[hparam] = float(hparams[hparam])
        elif 'filters' in hparam:
            hparams[hparam] = int(hparams[hparam])
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
        
        
        
        return avg_best, avg_best_ci, total_counts, num_sets_each_val, avg_acc_by_val, len(examples), len(examples[0])


def make_into_dict(avg_best, avg_best_ci, prop_counts, num_sets_each_val, avg_acc_by_val, num_samples, num_iters):
    info = {}
    info['avg_best'] = avg_best
    info['ci'] = avg_best_ci
    info['prop_counts'] = prop_counts
    info['num_sets_each_val'] = num_sets_each_val
    info['avg_acc_by_val'] = avg_acc_by_val
    info['num_samples'] = num_samples
    info['num_iters'] = num_iters
    return info

def make_scatter_and_hist():
    #import pdb; pdb.set_trace() 
    all_info = {}
    for space in ['reg_bad_lr', 'reg_half_bad_lr', 'reg']:
        all_info[space] = {}
        for dist in ['dpp', 'dpp_ham', 'dpp_rand']:
            things = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/20_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe={},spce={}.txt'.format(dist, space))
            all_info[space][dist] = make_into_dict(*things)
        
    counter = 0
    matplotlib.rcParams.update({'font.size': 4})
    for space in ['reg_bad_lr', 'reg_half_bad_lr', 'reg']:
        counter = counter + 1
        plt.subplot(2,3,counter)
        scatter_with_error_bars.add_scatter(all_info[space]['dpp'], all_info[space]['dpp_ham'], all_info[space]['dpp_rand'], space)

    for space in ['reg_bad_lr', 'reg_half_bad_lr', 'reg']:
        counter = counter + 1
        plt.subplot(2,3,counter)
        sample_histograms.add_hist(all_info[space]['dpp'], all_info[space]['dpp_ham'], 'num_sets_each_val')

    plt.savefig('plot_drafts/DEBUG_TRYING_ALL_THREE.pdf')
    

            
    
make_scatter_and_hist()
    
    






iters = '20'
space = 'reg_half_bad_lr'
model = 'cnn'
dist = '_ham'

dpp_avg_best, dpp_avg_best_ci, dpp_prop_counts, dpp_num_sets_each_val, dpp_avg_acc_by_val, dpp_num_samples, dpp_num_iters = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/{}_iter/stanford_sentiment_binary,mdl_tpe={},srch_tpe=dpp{},spce={}.txt'.format(iters, model, dist, space))

rand_avg_best, rand_avg_best_ci, rand_prop_counts, rand_num_sets_each_val, rand_avg_acc_by_val, rand_num_samples, rand_num_iters = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/{}_iter/stanford_sentiment_binary,mdl_tpe={},srch_tpe=dpp_rand,spce={}.txt'.format(iters, model, space))
#rand_avg_best2, rand_avg_best_ci2, rand_prop_counts2, rand_num_sets_each_val2, rand_avg_acc_by_val2, rand_num_samples2, rand_num_iters2 = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/{}_iter/stanford_sentiment_binary,mdl_tpe={},srch_tpe=dpp_rand,spce={}_SECOND_HUNDRED.txt'.format(iters, model, space))


#import sample_histograms
#import scatter_with_error_bars

#sample_histograms.plot_hists(dpp_num_sets_each_val, dpp_avg_acc_by_val, dpp_num_samples, dpp_num_iters, space + dist)

#sample_histograms.plot_hists(rand_num_sets_each_val2, rand_avg_acc_by_val2, rand_num_samples2, rand_num_iters2, space + '_random_SECOND_HUNDRED')

#scatter_with_error_bars.make_scatter(dpp_avg_best, dpp_avg_best_ci, rand_avg_best, rand_avg_best_ci, dpp_num_samples, rand_num_samples, dpp_num_iters, space + dist)



