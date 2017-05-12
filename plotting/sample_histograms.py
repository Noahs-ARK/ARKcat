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
    plt.twinx()
    plt.plot(pos, avg, color='r')
    plt.ylabel('average accuracy', color='r')
    plt.tick_params('y',colors='r')
    plt.ylim([0.5, 0.85])


def get_data(d):
    if d == '10_iter_bad_lr':
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


    return learning_rate_val, learning_rate_count, learning_rate_avg, l2_strength_val, l2_strength_avg, l2_strength_count, dropout_val, dropout_avg, dropout_count


#learning_rate_val, learning_rate_count, learning_rate_avg, l2_strength_val, l2_strength_avg, l2_strength_count, dropout_val, dropout_avg, dropout_count = get_data('10_iter_half_good_lr')


def plot_hists(props, avgs):

    matplotlib.rcParams.update({'font.size': 5})

    counter = 0
    for hparam in props:
        counter += 1
        vals = props[hparam]['vals']
        proportions = props[hparam]['prop']
        cur_avgs = avgs[hparam]
        plot_one_thing(vals, proportions, cur_avgs, counter, hparam)
    plt.tight_layout()
    plt.savefig('100_samples_of_10_as_multinomial_good_lr.pdf')

