import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sample_histograms
import scatter_with_error_bars
import os

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
        avg_best_ci.append(2.575*np.std(np.asarray(best_so_far[i]))/np.sqrt(len(examples)))
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
                
    
def frac_vals_in_each_set(examples):
    #hparam to fraction of time it's covered in one set
    hparam_to_num_values = {}
    hparam_to_num_poss_vals = {}
    for set_of_k in examples:
        seen_in_cur_set = {}
        for example in set_of_k:
            for hp in example[4]:

                # to add to seen_in_cur_set:
                if hp not in seen_in_cur_set:
                    seen_in_cur_set[hp] = {}
                v = example[4][hp]

                if v == 1.0:
                    continue

                if v not in seen_in_cur_set[hp]:
                    seen_in_cur_set[hp][v] = 0
                seen_in_cur_set[hp][v] += 1

                # to add hp to hparam_to_num_poss_vals:
                if hp not in hparam_to_num_poss_vals:
                    hparam_to_num_poss_vals[hp] = set()
                hparam_to_num_poss_vals[hp].add(v)
                    
                
        for hp in seen_in_cur_set:
            if hp not in hparam_to_num_values:
                hparam_to_num_values[hp] = []
            hparam_to_num_values[hp].append(len(seen_in_cur_set[hp]))
    # avg, std, med, max_seen, max, hp
    for hp in hparam_to_num_values:
        if max(hparam_to_num_values[hp]) != 1:
            to_print = ' & {} & {} & {} & {} & {} & {} & {} \\\\'.format(
                round(np.average(np.asarray(hparam_to_num_values[hp])),3),
                round(np.std(np.asarray(hparam_to_num_values[hp])),3),
                np.median(np.asarray(hparam_to_num_values[hp])),
                max(hparam_to_num_values[hp]),
                min(hparam_to_num_values[hp]),
                len(hparam_to_num_poss_vals[hp]),
                hp)
            print to_print
    
    

        


def get_avg_and_std_dev(file_loc):
    with open(file_loc) as f:
        lines = f.readlines()
        examples = extract_examples(lines)
        avg_best, avg_best_ci = compute_avg_best_so_far(examples)
        total_counts, num_sets_each_val = frac_set_has_hp_value(examples)
        frac_vals_in_each_set(examples)
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
    ordered_algos = ['dpp_rand',  'rand', 'tpe', 'dpp_ham', 'dpp' ]
    names = {'dpp_rand':'Uniform-D', 'dpp_ham':'$k$-DPP-Hamm', 'dpp':'$k$-DPP-Cos',
             'rand':'Uniform', 'tpe':'BO-TPE'}
    for space in ['reg_bad_lr', 'reg_half_bad_lr', 'reg']:
        all_info[space] = {}
        for dist in ordered_algos:
            file_name = '/homes/gws/jessedd/projects/ARKcat/plotting/results/20_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe={},spce={}.txt'.format(dist, space)
            
            if os.path.isfile(file_name):
                things = get_avg_and_std_dev(file_name)
                all_info[space][dist] = make_into_dict(*things)
                
        
    counter = 0
    matplotlib.rcParams.update({'font.size': 6})
    fig = plt.figure()
    #lines = ['-','--',':','-.','_']
    dash_list = [(5,2,20,2),(5,2,10,2),(5,2),(10,2),(3,2,2,2)] 
    for space in ['reg_bad_lr', 'reg_half_bad_lr', 'reg']:

        counter = counter + 1
        cur_ax = fig.add_subplot(1,3,counter, adjustable='box', aspect=100)
        data_for_scatter = {}
        dash_counter = 0
        for dist in ordered_algos:
            if dist in all_info[space]:
                data_for_scatter[dist] = {}
                data_for_scatter[dist]['avg_best'] = all_info[space][dist]['avg_best']
                data_for_scatter[dist]['dash'] = dash_list[dash_counter]
                dash_counter += 1        
        scatter_with_error_bars.add_scatter(data_for_scatter, ordered_algos, names, space, cur_ax)
    
        
    plt.tight_layout()
    plt.savefig('plot_drafts/DEBUG_THREE_MEDIAN.pdf',bbox_inches='tight')
    
def make_arch_scatter():
    all_info = {}
    for dist in ['dpp', 'dpp_ham', 'dpp_rand']:
        things = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/20_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe={},spce=arch.txt'.format(dist))
        all_info[dist] = make_into_dict(*things)
    fig = plt.figure()
    cur_ax = fig.add_subplot(1,1,1)
    scatter_with_error_bars.add_scatter(all_info['dpp'], all_info['dpp_ham'], all_info['dpp_rand'], 'arch', cur_ax)
    plt.tight_layout()
    plt.savefig('plot_drafts/DEBUG_ARCH.pdf',bbox_inches='tight')
            
    
def make_linear_scatter():
    all_info = {}
    for dist in ['dpp', 'dpp_rand']:
        things = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/50_iter/stanford_sentiment_binary,mdl_tpe=linear,srch_tpe={}.txt'.format(dist))
        all_info[dist] = make_into_dict(*things)


    #print all_info['dpp']['avg_best'][-1]
    #print all_info['dpp']['ci'][-1]
    #print all_info['dpp_rand']['avg_best'][-1]
    #print all_info['dpp_rand']['ci'][-1]
    #sys.exit(0)


    fig = plt.figure()
    cur_ax = fig.add_subplot(1,2,1)
    scatter_with_error_bars.add_scatter(all_info['dpp'], None, all_info['dpp_rand'], 'linear', cur_ax)
    


    all_info['dpp']['avg_best'] = all_info['dpp']['avg_best'][25:]
    all_info['dpp_rand']['avg_best'] = all_info['dpp_rand']['avg_best'][25:]
    

    cur_ax = fig.add_subplot(1,2,2)
    scatter_with_error_bars.add_scatter(all_info['dpp'], None, all_info['dpp_rand'], 'linear', cur_ax, .78, 0.795, 24)
    
    plt.tight_layout()
    plt.savefig('plot_drafts/DEBUG_LINEAR.pdf',bbox_inches='tight')


def make_empirical_hists():
    all_info = {}
    for space in ['reg_bad_lr', 'reg_half_bad_lr', 'reg']:
        things = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/20_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe=dpp_ham,spce={}.txt'.format(space))
        all_info[space] = make_into_dict(*things)
    new_all_info = {}
    
    
    new_all_info['all'] = {}
    new_all_info['all']['prop_counts'] = {}

    #import pdb; pdb.set_trace() 
    for space in ['reg_bad_lr', 'reg_half_bad_lr']:
        for hparam in all_info[space]['prop_counts']:
            if hparam not in new_all_info['all']['prop_counts']:
                new_all_info['all']['prop_counts'][hparam] = {}
            new_all_info['all']['prop_counts'][hparam]['vals'] = all_info[space]['prop_counts'][hparam]['vals']
            if 'prop' not in new_all_info['all']['prop_counts'][hparam]:
                new_all_info['all']['prop_counts'][hparam]['prop'] = [i*all_info[space]['num_samples']*all_info[space]['num_iters'] for i in all_info[space]['prop_counts'][hparam]['prop']]
            else:
                for i in range(len(new_all_info['all']['prop_counts'][hparam]['prop'])):
                    new_all_info['all']['prop_counts'][hparam]['prop'][i] += all_info[space]['prop_counts'][hparam]['prop'][i]*all_info[space]['num_samples']*all_info[space]['num_iters']


    total_samples = 0
    for space in ['reg_bad_lr', 'reg_half_bad_lr']:
        total_samples += all_info[space]['num_samples']*all_info[space]['num_iters']
    for hparam in new_all_info['all']['prop_counts']:
        new_all_info['all']['prop_counts'][hparam]['prop'] = [i/total_samples for i in new_all_info['all']['prop_counts'][hparam]['prop']]

    

    print new_all_info['all']['prop_counts']['reg_strength']['prop']
    del new_all_info['all']['prop_counts']['reg_strength']['prop'][-1]
    print new_all_info['all']['prop_counts']['reg_strength']['prop']

        

    

    dpp_info = sample_histograms.get_data()
    
    matplotlib.rcParams.update({'font.size': 7})
    fig = plt.figure()
    counter = 0
    for hparam in new_all_info['all']['prop_counts']:
        counter += 1
        cur_ax = fig.add_subplot(2,2,counter)
        if hparam == 'learning_rate':
            handles = sample_histograms.add_hist(dpp_info, new_all_info['all'], 'prop_counts', hparam, 'learning rate')
        elif hparam == 'reg_strength':
            handles = sample_histograms.add_hist(dpp_info, new_all_info['all'], 'prop_counts', hparam, 'regularization strength')
        elif hparam == 'dropout':
            handles = sample_histograms.add_hist(dpp_info, new_all_info['all'], 'prop_counts', hparam, 'dropout')
        elif hparam == 'regularizer':
            handles = sample_histograms.add_hist(dpp_info, new_all_info['all'], 'prop_counts', hparam, 'regularization')
    
    fig.legend(handles, ('k-DPP-Cos','k-DPP-Hamm'), loc='lower right')
    
    plt.tight_layout()
    plt.savefig('plot_drafts/DEBUG_FOUR_HISTS.pdf',bbox_inches='tight')
    

    #for hparam in all_info
    #new_all_info['all']['prop_counts'] = new_all_info['all']['prop_counts']*1.0/(all_info['reg_bad_lr']['num_samples'] + all_info['reg_half_bad_lr']['num_samples'])
    
    #print new_all_info['all']['prop_counts']
    
        
    
def print_best_acc():
    for space in ['arch']#['reg_bad_lr', 'reg_half_bad_lr', 'reg', 'arch']:
        for dist in ['dpp_rand', 'dpp_ham', 'dpp']:

            avg_best, avg_best_ci, prop_counts, num_sets_each_val, avg_acc_by_val, num_samples, num_iters = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/20_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe={},spce={}.txt'.format(dist, space))
            low_ci = round(100*avg_best[len(avg_best)-1]-avg_best_ci[len(avg_best_ci)-1],3)
            high_ci = round(100*avg_best[len(avg_best)-1]+avg_best_ci[len(avg_best_ci)-1],3)
            print '{}, {}: {} & \small ({},{}), {}'.format(space, dist, round(100*avg_best[len(avg_best)-1],3), low_ci, high_ci, avg_best_ci[len(avg_best_ci)-1])
    



#make_scatter_and_hist()
#make_arch_scatter()
#make_linear_scatter()
#make_empirical_hists()
print_best_acc()


#things = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/20_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe=dpp_rand,spce=reg_half_bad_lr.txt')
#things = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/20_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe=dpp_ham,spce=reg_half_bad_lr.txt')
#things = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/20_iter/stanford_sentiment_binary,mdl_tpe=cnn,srch_tpe=dpp,spce=reg_half_bad_lr.txt')
#all_info[space] = make_into_dict(*things)



#iters = '20'
#space = 'reg_half_bad_lr'
#model = 'cnn'
#dist = '_ham'

#dpp_avg_best, dpp_avg_best_ci, dpp_prop_counts, dpp_num_sets_each_val, dpp_avg_acc_by_val, dpp_num_samples, dpp_num_iters = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/{}_iter/stanford_sentiment_binary,mdl_tpe={},srch_tpe=dpp{},spce={}.txt'.format(iters, model, dist, space))

#rand_avg_best, rand_avg_best_ci, rand_prop_counts, rand_num_sets_each_val, rand_avg_acc_by_val, rand_num_samples, rand_num_iters = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/{}_iter/stanford_sentiment_binary,mdl_tpe={},srch_tpe=dpp_rand,spce={}.txt'.format(iters, model, space))
#rand_avg_best2, rand_avg_best_ci2, rand_prop_counts2, rand_num_sets_each_val2, rand_avg_acc_by_val2, rand_num_samples2, rand_num_iters2 = get_avg_and_std_dev('/homes/gws/jessedd/projects/ARKcat/plotting/results/{}_iter/stanford_sentiment_binary,mdl_tpe={},srch_tpe=dpp_rand,spce={}_SECOND_HUNDRED.txt'.format(iters, model, space))


#import sample_histograms
#import scatter_with_error_bars

#sample_histograms.plot_hists(dpp_num_sets_each_val, dpp_avg_acc_by_val, dpp_num_samples, dpp_num_iters, space + dist)

#sample_histograms.plot_hists(rand_num_sets_each_val2, rand_avg_acc_by_val2, rand_num_samples2, rand_num_iters2, space + '_random_SECOND_HUNDRED')

#scatter_with_error_bars.make_scatter(dpp_avg_best, dpp_avg_best_ci, rand_avg_best, rand_avg_best_ci, dpp_num_samples, rand_num_samples, dpp_num_iters, space + dist)



