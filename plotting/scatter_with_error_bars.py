import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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




def add_scatter(data, order, names, space, cur_ax, ylim_lower=None, ylim_upper=None, last_few=None):
    x = range(1,len(data['dpp']['avg_best'])+1)
    if last_few is not None:
        x = [xi + last_few for xi in x]
    diff_x = {}
    counter = 0
    width = .3
    for algo in order:
        counter += 1
        #diff_x[algo] = [(width*1.0/len(data)) * counter + x_i for x_i in x]
        diff_x[algo] = x
    #dpp_x = [i-.15 for i in x]
    #ham_x = x
    #rand_x = [i+.15 for i in x]

    if space != 'arch':
        cur_ax.set_ylim([.545,.83])
    if ylim_lower is not None:
        cur_ax.set_ylim([ylim_lower,ylim_upper])

    #cur_ax.errorbar(rand_x, rand_info['avg_best'], yerr=rand_info['ci'], fmt=',',elinewidth=.5, markersize=2)
    #cur_ax.errorbar(dpp_x, dpp_info['avg_best'], yerr=dpp_info['ci'], fmt=',',elinewidth=.5,markersize=2)
    #cur_ax.errorbar(ham_x, ham_info['avg_best'], yerr=ham_info['ci'], fmt=',',elinewidth=.5, markersize=2)
    for algo in order:
        if algo in data:
            cur_ax.plot(diff_x[algo], data[algo]['avg_best'], dashes=data[algo]['dash'], linewidth=1)
    #cur_ax.plot(rand_x, rand_info['avg_best'], '-', linewidth=1)
    #if ham_info is not None:
    #    cur_ax.plot(ham_x, ham_info['avg_best'], '--', linewidth=1)
    #cur_ax.plot(dpp_x, dpp_info['avg_best'],':', linewidth=1)

    if space == 'reg_bad_lr':
        cur_ax.set_title("Hard learning rate".format(space))
    elif space == 'reg_half_bad_lr':
        cur_ax.set_title("Medium learning rate".format(space))
    elif space == 'reg':
        cur_ax.set_title("Easy learning rate".format(space))
    
    cur_ax.legend([names[name] for name in order if name in data], loc='lower right')
    #if ham_info is not None:
    #    cur_ax.legend(["Uniform", "k-DPP-Hamm", "k-DPP-Cos"],loc='lower right')
    #else:
    #    cur_ax.legend(["Uniform", "k-DPP-Cos"],loc='lower right')

    cur_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))


    #cur_ax.locator_params(axis='x', nticks=11)


    #num_vals = len(x)
    #pos = np.arange(num_vals)
    #ticks = []
    #for i in range(10):
    #    ticks.append(str(i*2 + 1))
    #    ticks.append('')
    #print ticks
    #cur_ax.set_xticks(ticks)
