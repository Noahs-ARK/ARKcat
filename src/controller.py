import os, sys
import codecs
import datetime
import time

import cPickle as pickle

import argparse
import numpy as np

import classify_test
import space_manager
from run_spearmint import spearmint_main
from run_hyperopt import hyperopt_main

import cProfile, pstats

#Note on running: use cbr.sh or cbr_long.sh for bayesopt; run_file.sh for reading file

#Note on file system: bayesopt cnn models save to model_dir/iter #/, while evaluated
#lines from file save to model_dir/.

def call_experiment(args):
    #in case we want to debug the BO algorithms
    #import pdb; pdb.set_trace()
    debug_mode = True
    if debug_mode:
        import random
        from hyperopt import STATUS_OK
        return {'loss':random.random(), 'status': STATUS_OK, 'duration': 10}

    start_time = time.time()
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
    result['duration'] = time.time() - start_time
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


def main(args):
    print("Made it to the start of main!")
    start_time = time.time()
    print("the time at the start: " + str(start_time))
    set_globals(args)
    if "spearmint" in args['algorithm']:
        spearmint_main(num_models, model_types, search_space, max_iter, args, call_experiment, model_dir)
    else:
        hyperopt_main(num_models, model_types, search_space, max_iter, args, call_experiment)


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
    parser.add_argument('batch_size', type=int, help='the batch size for running BO algorithms')
    print vars(parser.parse_args(sys.argv[1:]))
    main(vars(parser.parse_args(sys.argv[1:])))
    
