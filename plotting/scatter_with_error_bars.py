import numpy as np
import matplotlib.pyplot as plt


def make_scatter(dpp_y, dpp_err, rand_y, rand_err, dpp_num_samples, rand_num_samples, dpp_num_iters, space):

    x = range(1,len(dpp_y)+1)
    dpp_x = [i-.1 for i in x]
    rand_x = [i+.1 for i in x]

    # First illustrate basic pyplot interface, using defaults where possible.
    plt.figure()
    plt.errorbar(dpp_x, dpp_y, yerr=dpp_err, fmt='o',elinewidth=.5,markersize=2)
    plt.errorbar(rand_x, rand_y, yerr=rand_err, fmt='s',elinewidth=.5, markersize=2)
    plt.title("Mean best dev acc after K iterations, with 95% CI")
    plt.legend(["DPP","Random"],loc='lower right')
    
    plt.savefig('plot_drafts/cnn_iter={}_dppRestarts={}_randRestarts={}_space={}.pdf'.format(dpp_num_iters, dpp_num_samples, rand_num_samples, space))




def add_scatter(dpp_info, ham_info, rand_info, space):
    x = range(1,len(dpp_info['avg_best'])+1)
    dpp_x = [i-.15 for i in x]
    ham_x = x
    rand_x = [i+.15 for i in x]
    
    plt.ylim([.53,.83])
    #plt.errorbar(dpp_x, dpp_info['avg_best'], yerr=dpp_info['ci'], fmt=',',elinewidth=.5,markersize=2)
    #plt.errorbar(ham_x, ham_info['avg_best'], yerr=ham_info['ci'], fmt=',',elinewidth=.5, markersize=2)
    #plt.errorbar(rand_x, rand_info['avg_best'], yerr=rand_info['ci'], fmt=',',elinewidth=.5, markersize=2)
    plt.plot(rand_x, rand_info['avg_best'])
    plt.plot(ham_x, ham_info['avg_best'])

    plt.plot(dpp_x, dpp_info['avg_best'])
    plt.title("Mean best dev acc after K iterations on {}".format(space))
    plt.legend(["Random", "DPP-Hamm", "DPP"],loc='lower right')
