import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_one_thing(val, prop, avg, subplot_num, xlabel):
    pos = np.arange(len(val))

    plt.subplot(2,2,subplot_num)
    plt.bar(pos, prop, align='center')
    plt.xlabel(xlabel + ' values')
    plt.ylabel('proportion of times observed', color='b')
    plt.tick_params('y',colors='b')
    plt.xticks(pos, val, rotation=70)
    #plt.ylim([.9,1])
    plt.twinx()
    plt.plot(pos, avg, color='r')
    plt.ylabel('average accuracy', color='r')
    plt.tick_params('y',colors='r')
    plt.ylim([0.5, 0.85])
    
def add_hist(dpp_info, ham_info, prop_to_use, hparam, xlabel):
    #print dpp_info
    #print dpp_info['prop_counts']
    #print dpp_info['prop_counts'][hparam]['vals']
    num_vals = len(dpp_info['prop_counts'][hparam]['vals'])
    pos = np.arange(num_vals)
    #num_vals = num_vals * 1.0
    #prob_point_not_covered = 1-((num_vals-1.0)/num_vals)**dpp_info['num_iters']
    
    dpp_prop_counts = dpp_info[prop_to_use][hparam]['prop']
    ham_prop_counts = ham_info[prop_to_use][hparam]['prop']
    print len(ham_prop_counts)
    print ham_prop_counts
    #rand_prop_counts = [prob_point_not_covered] * len(dpp_prop_counts)
    
    h1 = plt.bar([x - 0.175 for x in pos], dpp_prop_counts, 0.35, align='center')
    h2 = plt.bar([x + 0.175 for x in pos], ham_prop_counts, 0.35, align='center', color='c')
    #plt.bar([x + 0.2 for x in pos], rand_prop_counts, width=0.2,color='blue', align='center')
    plt.xticks(pos, dpp_info['prop_counts'][hparam]['vals'], rotation=70)
    plt.ylabel('proportion of times observed', color = 'b')
    plt.tick_params('y',colors='b')
    plt.xlabel(xlabel + ' values')

    plt.twinx()
    # make average accuracy:
    #avg_acc = []
    #for i in range(len(dpp_info['avg_acc_by_val']['learning_rate'])):
    #    avg_acc.append((dpp_info['avg_acc_by_val']['learning_rate'][i] + ham_info['avg_acc_by_val']['learning_rate'][i])/2.0)
    plt.plot(pos, dpp_info['avg_acc_by_val'][hparam], color='r')
    plt.tick_params('y',colors='r')
    plt.ylabel('average accuracy', color='r')
    plt.ylim(0.5,0.85)
    
    return [h1,h2]
    
    
    
    
    


def get_data():
    # note: these were made from samples with k = 10!
    learning_rate_val = [0.0067, 0.0131, 0.0256, 0.0498, 0.097, 0.1889, 0.3679, 0.7165, 1.3956, 2.7183, 5.2945, 10.3123, 20.0855, 39.1213, 76.1979, 148.4132]
    learning_rate_count = [343, 332, 287, 268, 258, 247, 222, 236, 251, 242, 234, 244, 269, 292, 265, 410]
    learning_rate_avg = [0.763,0.715,0.643,0.523,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52,0.52]
        
    l2_strength_val = [-5.0, -4.7333, -4.4667, -4.2, -3.9333, -3.6667, -3.4, -3.1333, -2.8667, -2.6, -2.3333, -2.0667, -1.8, -1.5333, -1.2667, -1.0]
    l2_strength_val = [round(np.exp(x),4) for x in l2_strength_val]
    l2_strength_avg = [0.559,0.561,0.56,0.562,0.563,0.554,0.554,0.557,0.561,0.555,0.555,0.561,0.554,0.548,0.534,0.52]
    l2_strength_count = [294, 270, 241, 223, 212, 217, 202, 170, 202, 215, 218, 246, 214, 220, 260, 420]
        
    dropout_val = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    dropout_avg = [0.55,0.555,0.552,0.553,0.556,0.553,0.555,0.561,0.558,0.547,0.555,0.562,0.544,0.541,0.557]
    dropout_count = [346, 358, 312, 295, 287, 314, 243, 258, 256, 309, 283, 255, 262, 317, 305]

    dpp_info = {}
    dpp_info['avg_acc_by_val'] = {}
    dpp_info['prop_counts'] = {}

    dpp_info['prop_counts']['learning_rate'] = {}
    dpp_info['prop_counts']['learning_rate']['prop'] = [i*1.0 / 4400 for i in learning_rate_count]
    dpp_info['prop_counts']['learning_rate']['vals'] = learning_rate_val
    dpp_info['avg_acc_by_val']['learning_rate'] = learning_rate_avg

    dpp_info['prop_counts']['reg_strength'] = {}
    dpp_info['prop_counts']['reg_strength']['prop'] = [i*1.0 / 4400 for i in l2_strength_count]
    dpp_info['prop_counts']['reg_strength']['vals'] = l2_strength_val
    dpp_info['avg_acc_by_val']['reg_strength'] = l2_strength_avg
    
    dpp_info['prop_counts']['dropout'] = {}
    dpp_info['prop_counts']['dropout']['prop'] = [i*1.0 / 4400 for i in dropout_count]
    dpp_info['prop_counts']['dropout']['vals'] = dropout_val
    dpp_info['avg_acc_by_val']['dropout'] = dropout_avg

    dpp_info['prop_counts']['regularizer'] = {}
    dpp_info['prop_counts']['regularizer']['prop'] = [1.0*576/4400, 3824*1.0/4400]
    dpp_info['prop_counts']['regularizer']['vals'] = ['None', 'L2']
    dpp_info['avg_acc_by_val']['regularizer'] = [0.564, 0.552]

    return dpp_info







#dropout
#reg_strength
#regularizer



    return learning_rate_val, learning_rate_count, learning_rate_avg, l2_strength_val, l2_strength_avg, l2_strength_count, dropout_val, dropout_avg, dropout_count


#learning_rate_val, learning_rate_count, learning_rate_avg, l2_strength_val, l2_strength_avg, l2_strength_count, dropout_val, dropout_avg, dropout_count = get_data('10_iter_half_good_lr')


def plot_hists(props, avgs, num_samples, num_iters, space):

    matplotlib.rcParams.update({'font.size': 5})

    counter = 0
    for hparam in props:
        counter += 1
        vals = props[hparam]['vals']
        proportions = props[hparam]['prop']
        cur_avgs = avgs[hparam]
        plot_one_thing(vals, proportions, cur_avgs, counter, hparam)
    plt.tight_layout()
    plt.savefig('plot_drafts/{}_samples_of_{}_propCount_space={}.pdf'.format(num_samples, num_iters, space))

