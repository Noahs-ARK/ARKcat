import os, sys
import codecs
import datetime
import cPickle as pickle
import Queue as queue

from optparse import OptionParser
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, space_eval

import classify_test

from run import *

def call_experiment(args):
    global trial_num
    trial_num = trial_num + 1

    feats_and_args = {}
    all_description = []
    for i in range(num_models):
        feature_list, description, kwargs = wrangle_params(args, str(i))
        all_description = all_description + description
        feats_and_args[i] = {'feats':feature_list, 'params':kwargs}


    result = classify_test.classify(train_data_filename, train_label_filename, dev_data_filename,
                                    dev_label_filename, train_feature_dir, dev_feature_dir,
                                    model_dir, feats_and_args, folds=num_folds)



    with codecs.open(log_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + ' '.join(all_description) +
                          '\t' + str(-result['loss']) + '\n')
    save_model(result)

    print("\nFinished iteration " + str(trial_num) + ".\n\n\n")
    sys.stdout.flush()
    return result

def grid_search(model_types):
    feature_selector = cnn_feature_selector()
    hyperparam_grid: {
            'model_' + model_num: 'CNN',
            'delta_' + model_num: (feature_selector['delta_']),
            'flex_' + model_num: [(False, 0.0),
                (True, [.15, .3])],
            'filters_' + model_num: list(feature_selector['filters_']),
            'kernel_size_1_' + model_num: list(feature_selector['kernels_']),
            'kernel_size_2_' + model_num: list(feature_selector['kernels_']),
            'kernel_size_3_' + model_num: list(feature_selector['kernels_']),
            'dropout_' + model_num: [.25, .75],
            'batch_size_' + model_num: list(feature_selector['batch_size_']),
            # iden, relu, and tanh
            'activation_fn_' + model_num: feature_selector['activation_fn_'],
            #none, clipped, or penalized
            'regularizer_cnn_' + model_num: [
                (None, 0.0),
                ('l2', feature_selector['l2_']),
                ('l2_clip', feature_selector['l2_clip_'])
            ]),
            'learning_rate_' + model_num: [.00075, .0015])
    }

def main():
    print("Made it to the start of main!")
    set_globals()
    trials = Trials()
    space = grid_search(model_types)
    best = fmin(call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=max_iter,
                trials=trials)

    print space_eval(space, best)
    printing_best(trials)


if __name__ == '__main__':
    main()
