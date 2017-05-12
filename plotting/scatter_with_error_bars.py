import numpy as np
import matplotlib.pyplot as plt


def make_scatter(dpp_y, dpp_err, rand_y, rand_err):

    x = range(1,len(dpp_y)+1)
    dpp_x = [i-.1 for i in x]
    rand_x = [i+.1 for i in x]

    # First illustrate basic pyplot interface, using defaults where possible.
    plt.figure()
    plt.errorbar(dpp_x, dpp_y, yerr=dpp_err, fmt='o',elinewidth=.5,markersize=2)
    plt.errorbar(rand_x, rand_y, yerr=rand_err, fmt='s',elinewidth=.5, markersize=2)
    plt.title("Mean best dev acc after K iterations, with 95% CI")
    plt.legend(["DPP","Random"],loc='lower right')
    
    plt.savefig('cnn_iter=10_dppRestarts=100_randRestarts=100_space=reg.pdf')
