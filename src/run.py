import os, sys
import codecs
import datetime
import time

import cPickle as pickle
import Queue as queue

import argparse
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, space_eval, rand, anneal
from hyperopt import dpp, dpp_random, sample_hparam_space

import classify_test
import space_manager
from grid_search import *

import cProfile, pstats

#Note on running: use cbr.sh or cbr_long.sh for bayesopt; run_file.sh for reading file

#Note on file system: bayesopt cnn models save to model_dir/iter #/, while evaluated
#lines from file save to model_dir/.

def call_experiment(args):
    #in case we want to debug the BO algorithms
    #import pdb; pdb.set_trace()
    debug_mode = False
    if debug_mode:
        import random
        from hyperopt import STATUS_OK
        return {'loss':random.random(), 'status': STATUS_OK}

    
    global trial_num
    trial_num = trial_num + 1
    print model_dir
    if line_num: #if reading from file
        print 'file'
        new_model_dir = model_dir
    else:
        new_model_dir = model_dir + str(trial_num) + '/'
        os.makedirs(new_model_dir)
    print new_model_dir
    feats_and_args = {}
    all_description = []
    for i in range(num_models):
        feature_list, description, kwargs = wrangle_params(args, str(i))
        all_description = all_description + description
        feats_and_args[i] = {'feats':feature_list, 'params':kwargs}


    result = classify_test.classify(train_data_filename, train_label_filename, dev_data_filename,
                                    dev_label_filename, train_feature_dir, dev_feature_dir,
                                    new_model_dir, word2vec_filename, feats_and_args, folds=num_folds)



    with codecs.open(log_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + ' '.join(all_description) +
                          '\t' + str(-result['loss']) + '\n')
    save_model(result)

    print("\nFinished iteration " + str(trial_num) + ".\n\n\n")
    sys.stdout.flush()
    return result

#have to edit features--cnn won't take idf, for example
def wrangle_params(args, model_num):
    kwargs = {}

    print('')
    print('the args:')
    print(args)

    model = args['model_' + model_num]['model_type_' + model_num]
    kwargs['model_type'] = model
    if model == 'LR':
        kwargs['regularizer'] = args['model_' + model_num]['regularizer_lr_' + model_num][0]
        kwargs['alpha'] = args['model_' + model_num]['regularizer_lr_' + model_num][1]
        kwargs['converg_tol'] = args['model_' + model_num]['converg_tol_' + model_num]
    elif  model == 'XGBoost':
        kwargs['eta'] = args['model_' + model_num]['eta_' + model_num]
        kwargs['gamma'] = args['model_' + model_num]['gamma_' + model_num]
        kwargs['max_depth'] = int(args['model_' + model_num]['max_depth_' + model_num])
        kwargs['min_child_weight'] = args['model_' + model_num]['min_child_weight_' + model_num]
        kwargs['max_delta_step'] = args['model_' + model_num]['max_delta_step_' + model_num]
        kwargs['subsample'] = args['model_' + model_num]['subsample_' + model_num]
        kwargs['regularizer'] = args['model_' + model_num]['regularizer_xgb_' + model_num][0]
        kwargs['reg_strength'] = args['model_' + model_num]['regularizer_xgb_' + model_num][1]
        kwargs['num_round'] = int(args['model_' + model_num]['num_round_' + model_num])
    elif model == 'CNN':
        kwargs['word_vector_init'] = args['model_' + model_num]['word_vectors_' + model_num][0]
        kwargs['word_vector_update'] = args['model_' + model_num]['word_vectors_' + model_num][1]
        kwargs['delta'] = args['model_' + model_num]['delta_' + model_num]
        kwargs['flex'] = True #DEBUGGING can search over this in space_manager
        kwargs['flex_amt'] = (args['model_' + model_num]['flex_amt_' + model_num])
        kwargs['kernel_size'] = int(args['model_' + model_num]['kernel_size_' + model_num])
        kwargs['kernel_increment'] = int(args['model_' + model_num]['kernel_increment_' + model_num])
        kwargs['kernel_num'] = int(args['model_' + model_num]['kernel_num_' + model_num])
        kwargs['filters'] = int(args['model_' + model_num]['filters_' + model_num])
        kwargs['dropout'] = args['model_' + model_num]['dropout_' + model_num]
        kwargs['batch_size'] = int(args['model_' + model_num]['batch_size_' + model_num])
        kwargs['activation_fn'] = args['model_' + model_num]['activation_fn_' + model_num]
        kwargs['regularizer'] = args['model_' + model_num]['regularizer_cnn_' + model_num][0]
        kwargs['reg_strength'] = args['model_' + model_num]['regularizer_cnn_' + model_num][1]
        kwargs['learning_rate'] = args['model_' + model_num]['learning_rate_' + model_num]

    features = {}
    features['ngram_range'] = args['features_' + model_num]['nmin_to_max_' + model_num]
    features['binary'] = args['features_' + model_num]['binary_' + model_num]
    features['use_idf'] = args['features_' + model_num]['use_idf_' + model_num]
    features['stop_words'] = args['features_' + model_num]['st_wrd_' + model_num]

    print kwargs
    print features
    description = [str(k) + '=' + str(v) for (k, v) in kwargs.items()]
    description[0] = description[0] + ',' + [str(k) + '=' + str(v) for (k, v) in features.items()][0]
    return features, description, kwargs


def save_model(result):
    model = result['model']
    feature_list = result['model'].feats_and_params[0]['feats']
    model_hyperparams = result['model'].feats_and_params[0]['params']
    #STUPID FILENAMES TOO LONG
    short_name = {'model_type':'mdl', 'regularizer':'rg', 'converg_tol':'cvrg','alpha':'alpha',
                  'eta':'eta', 'gamma':'gamma', 'max_depth':'dpth', 'min_child_weight':'mn_wght',
                  'max_delta_step':'mx_stp', 'subsample':'smpl', 'reg_strength':'rg_str',
                  'num_round':'rnds', 'lambda':'lmbda', 'ngram_range':'ngrms', 'binary':'bnry',
                  'use_idf':'idf', 'stop_words':'st_wrd', 'word_vector_init':'wv_init',
                  'word_vector_update':'upd', 'delta':'delta', 'flex':'flex', 'flex_amt':'flex_amt',
                  'kernel_size':'ks', 'kernel_increment':'ki', 'kernel_num':'kn',
                  'filters':'fltrs', 'dropout':'drop', 'batch_size':'batch',
                  'activation_fn':'actvn', 'regularizer':'rg', 'reg_strength':'rg_str',
                  'learning_rate':'learn_rt'}

    # to save the model after each iteration
    feature_string = ''
    for feat, value in feature_list.items():
        feature_string = feature_string + short_name[feat] + '=' + str(value) + ';'
    for hparam in model_hyperparams:
        cur_hparam = None
        #DEBUGGING
        if hparam == 'folds' or hparam == 'model_num':
            continue
        if isinstance(model_hyperparams[hparam], float):
            cur_hparam = str(round(model_hyperparams[hparam]*1000)/1000)
        else:
            cur_hparam = str(model_hyperparams[hparam])
        try:
            feature_string = feature_string + short_name[hparam] + '=' + cur_hparam + ';'
        except:
            sys.stderr.write('\nnot found ' + hparam)
    feature_string = feature_string[:-1]
    pickle.dump([trial_num, train_feature_dir, result], open(model_dir + str(trial_num) + '_' +
                                                             feature_string + '.model', 'wb'))

#sets the global variables, including params passed as args
def set_globals(args):
    for k, v in args.iteritems():
        try:
            if len(v) == 1:
                args[k] = v[0]
        except TypeError: #v is None
            pass
    global train_data_filename, train_label_filename, dev_data_filename, dev_label_filename
    global output_dir, train_feature_dir, dev_feature_dir, model_dir, word2vec_filename, log_filename
    global trial_num, max_iter, num_folds, num_models, model_types, search_space, model_path, line_num
    train_data_filename = args['dataset'] + 'train.data'
    train_label_filename = args['dataset'] + 'train.labels'
    dev_data_filename = args['dataset'] + 'dev.data'
    dev_label_filename = args['dataset'] + 'dev.labels'
    print train_data_filename, dev_data_filename
    word2vec_filename = args['word2vec_filename']
    output_dir = args['output_dir']
    num_folds = args['num_folds']
    train_feature_dir = output_dir + '/train_features/'
    dev_feature_dir = output_dir + '/dev_train_features/'
    model_dir = output_dir + '/saved_models/'

    trial_num = 0
    if args['run_bayesopt']:
        num_models = 1
        model_types = [args['run_bayesopt'][0]]
        search_space = args['run_bayesopt'][1]
        max_iter = int(args['run_bayesopt'][2])
        line_num = None
    else: #we're loading from a file
        num_models = 1
        model_path = args['load_file'][0] #path to the file with saved hparams to try
        line_num = int(args['load_file'][1]) #line number in the file
        trial_num = line_num
        # model_dir += str(line_num) + '/'

    for directory in [output_dir, train_feature_dir, dev_feature_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    log_filename = os.path.join(output_dir, 'log.txt')

    print dev_data_filename

    with open(log_filename, 'w') as logfile:
        logfile.write(','.join([train_data_filename, train_label_filename, dev_data_filename,
                                dev_label_filename, train_feature_dir, dev_feature_dir, output_dir]) + '\n')

def printing_best(trials):
    priority_q = queue.PriorityQueue()
    losses = trials.losses()
    for i in range(len(losses)):
        priority_q.put((losses[i], i))
    print('top losses and settings: ')
    for i in range(0,min(3,max_iter)):
        index = priority_q.get()[1]
        print(losses[index])
        print(trials.trials[index]['misc']['vals'])
        print('')
    print('')

def set_discretize_num(trials):
    if search_space == 'arch':
        trials.discretize_num = 4
    elif 'reg' in search_space:
        trials.discretize_num = 15
    elif 'debug' in search_space:
        trials.discretize_num = None
    else:
        raise ValueError("you tried to use " + search_space + " as a search space, but we don't know how many "+
                         "values we should discretize to (for the dpp)")

def main(args):
    print("Made it to the start of main!")
    print("the time at the start: " + str(time.time()))
    set_globals(args)
    trials = Trials()
    trials.discretize_space = True
    # a hacky solution to pass parameters to hyperopt
    if trials.discretize_space:
        set_discretize_num(trials)
        # DEBUGGING
        trials.discretize_num = 5
    if args['run_bayesopt']:
        space = space_manager.get_space(num_models, model_types, search_space)
        if args['algorithm'] == "bayes_opt":
            algorithm = tpe.suggest
        elif args['algorithm'] == "random":
            algorithm = rand.suggest
        elif args['algorithm'] == "anneal":
            algorithm = anneal.suggest
        elif args['algorithm'] == "dpp_cos":
            trials.dpp_dist = "cos"
            algorithm = dpp.suggest
        elif args['algorithm'] == "dpp_ham":
            trials.dpp_dist = "ham"
            algorithm = dpp.suggest
        elif args['algorithm'] == "dpp_l2":
            trials.dpp_dist = "l2"
            algorithm = dpp.suggest
        elif args['algorithm'] == 'mixed_dpp_rbf':
            trials.dpp_dist = "rbf"
            algorithm = dpp.suggest
            trials.discretize_space = False
        elif args['algorithm'] == "dpp_random":
            algorithm = dpp_random.suggest
        else:
            raise NameError("Unknown algorithm for search")

        #DEBUGGING: this is for profiling. it prints where the program has spent the most time
        #profile = cProfile.Profile()
        #import pdb; pdb.set_trace()
        #tmp = sample_hparam_space(space, algorithm, max_iter, 'l2', True, 15)
        try:
            #profile.enable()

            best = fmin(call_experiment,
                        space=space,
                        algo=algorithm,
                        max_evals=max_iter,
                        trials=trials)
            #profile.disable()
        finally:
            #profile = pstats.Stats(profile).sort_stats('cumulative')
            #profile.print_stats()
            print('')

        print space_eval(space, best)
        printing_best(trials)
    #loading models from file
    else:
        with open(model_path) as f:
            for i in range(line_num - 1):
                f.readline()

            space = eval(f.readline())
            best = call_experiment(space)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='need to write one')
    parser.add_argument('dataset', nargs=1, type=str, help='')
    parser.add_argument('word2vec_filename', nargs=1, type=str, help='')
    parser.add_argument('output_dir', nargs=1, type=str, help='')
    parser.add_argument('num_folds', nargs=1, type=int, help='')
    parser.add_argument('algorithm', nargs=1, type=str, 
                        help='the algorithm to use. must be random, bayes_opt, dpp')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-b', nargs=3, type=str, dest='run_bayesopt',
                        help='model types, search space, number of iters')
    group.add_argument('-f', nargs=2, type=str, dest='load_file')
    print vars(parser.parse_args(sys.argv[1:]))
    main(vars(parser.parse_args(sys.argv[1:])))
