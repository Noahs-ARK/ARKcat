import os, sys
import codecs
import datetime
import cPickle as pickle
import Queue as queue

from optparse import OptionParser
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, space_eval

import classify_test
import space_manager

data_filename = None
label_filename = None
feature_dir = None
output_dir = None
log_filename = None

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
                                    feats_and_args)



    with codecs.open(log_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + ' '.join(all_description) +
                          '\t' + str(-result['loss']) + '\n')
    save_model(result)

    print("\nFinished iteration " + str(trial_num) + ".\n\n\n")
    sys.stdout.flush()
    return result

def wrangle_params(args, model_num):
    kwargs = {}

    # DEBUGGING 
    # WARNING THIS IS A HACK! Should pass this is as a param
    kwargs['folds'] = 0

    print('')
    print('the args:')
    print(args)


    model = args['model_' + model_num]['model_' + model_num]
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
        

    features = {}
    features['ngram_range'] = args['features_' + model_num]['nmin_to_max_' + model_num]
    features['binary'] = args['features_' + model_num]['binary_' + model_num]
    features['use_idf'] = args['features_' + model_num]['use_idf_' + model_num]
    features['stop_words'] = args['features_' + model_num]['st_wrd_' + model_num]

        
    #DEBUGGING
    #kwargs['regularizer'] = 'l2'
    #kwargs['alpha'] = 10
    #kwargs['converg_tol'] = .098
    #END DEBUGGING

    print kwargs
    print features
    description = [str(k) + '=' + str(v) for (k, v) in kwargs.items()]
    description[0] = description[0] + ',' + [str(k) + '=' + str(v) for (k, v) in features.items()][0]
    return features, description, kwargs
    

def save_model(result):
    model = result['model']
    feature_list = result['model'].feats_and_params[0]['feats']
    model_hyperparams = result['model'].feats_and_params[0]['params']
    #STUPID FILENAMES TOO LING
    short_name = {'model_type':'mdl', 'regularizer':'rg', 'converg_tol':'cvrg','alpha':'alpha',
                  'eta':'eta', 'gamma':'gamma', 'max_depth':'dpth', 'min_child_weight':'mn_wght', 
                  'max_delta_step':'mx_stp', 'subsample':'smpl', 'reg_strength':'rg_str', 
                  'num_round':'rnds', 'lambda':'lmbda', 'ngram_range':'ngrms', 'binary':'bnry',
                  'use_idf':'idf', 'stop_words':'st_wrd'}


    # to save the model after each iteration
    feature_string = ''
    for feat, value in feature_list.items():
        feature_string = feature_string + short_name[feat] + '=' + str(value) + ';'
    for hparam in model_hyperparams:
        cur_hparam = None
        if hparam == 'folds':
            continue
        if isinstance(model_hyperparams[hparam], float):
            cur_hparam = str(round(model_hyperparams[hparam]*1000)/1000) 
        else:
            cur_hparam = str(model_hyperparams[hparam])
        feature_string = feature_string + short_name[hparam] + '=' + cur_hparam + ';'
    feature_string = feature_string[:-1]
    pickle.dump([trial_num, train_feature_dir, result], open(model_dir + str(trial_num) + '_' + 
                                                             feature_string + '.model', 'wb'))
    


#sets the global variables, including params passed as args
def set_globals():
    usage = "%prog train_text.json train_labels.csv dev_text.json dev_labels.csv output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='max_iter', default=30,
                      help='Maximum iterations of Bayesian optimization; default=%default')

    (options, args) = parser.parse_args()

    global train_data_filename, train_label_filename, dev_data_filename, dev_label_filename
    global output_dir, train_feature_dir, dev_feature_dir, model_dir, log_filename, trial_num, max_iter
    global num_models, model_types
    
    train_data_filename = args[0]
    train_label_filename = args[1]
    dev_data_filename = args[2]
    dev_label_filename = args[3]
    output_dir = args[4]
    num_models = int(args[5])
    model_types = args[6].split('-')
    

    train_feature_dir = output_dir + '/train_features/'
    dev_feature_dir = output_dir + '/dev_train_features/'
    model_dir = output_dir + '/saved_models/'
    
    trial_num = 0
    max_iter = int(options.max_iter)
    
    for directory in [output_dir, train_feature_dir, dev_feature_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    log_filename = os.path.join(output_dir, 'log.txt')

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
        

def main():
    set_globals()
    trials = Trials()
    space = space_manager.get_space(num_models, model_types)
    best = fmin(call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=max_iter,
                trials=trials)
    
    print space_eval(space, best)
    printing_best(trials)


if __name__ == '__main__':
    main()
