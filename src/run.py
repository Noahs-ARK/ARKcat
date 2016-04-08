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
    all_feat_list = []
    all_description = []
    for i in range(num_models):
        feature_list, description, kwargs = wrangle_params(args, str(i))
        all_feat_list = all_feat_list + feature_list
        all_description = all_description + description
        feats_and_args[i] = {'feats':feature_list, 'params':kwargs}

    ### TOTAL HACK DEBUGGING
    
#    feature_list = ['ngrams,n=1,transform=None,min_df=1']

    ### END TOTAL HACK DEBUGGING
    
    result = classify_test.classify(train_data_filename, train_label_filename, dev_data_filename, 
                                    dev_label_filename, train_feature_dir, dev_feature_dir, 
                                    feats_and_args)
    with codecs.open(log_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + ' '.join(all_feat_list) + '\t' + ' '.join(all_description) +
                          '\t' + str(-result['loss']) + '\n')
    save_model(result)

    print("\nFinished iteration " + str(trial_num) + ".\n\n\n")
    return result

def wrangle_params(args, model_num):
    kwargs = {}

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
        

    feature_list = []
    unigrams = 'ngrams,n=1' + \
               ',transform=' + args['features_' + model_num]['unigrams_' + model_num]['transform_' + model_num] + \
               ',min_df=' + str(args['features_' + model_num]['unigrams_' + model_num]['min_df_' + model_num])
    feature_list.append(unigrams)
    if args['features_' + model_num]['bigrams_' + model_num]['use_' + model_num]:
        bigrams = 'ngrams,n=2' + \
                  ',transform=' + args['features_' + model_num]['bigrams_' + model_num]['transform_' + model_num] + \
                  ',min_df=' + str(args['features_' + model_num]['bigrams_' + model_num]['min_df_' + model_num])
        feature_list.append(bigrams)

    print feature_list
    description = [str(k) + '=' + str(v) for (k, v) in kwargs.items()]
    return feature_list, description, kwargs
    

def save_model(result):
    model = result['model']
    feature_list = result['model'].feats_and_params[0]['feats']
    model_hyperparams = result['model'].feats_and_params[0]['params']
    #STUPID FILENAMES TOO LING
    short_name = {'model_type':'mdl', 'regularizer':'rg', 'converg_tol':'cvrg','alpha':'alpha',
                  'eta':'eta', 'gamma':'gamma', 'max_depth':'dpth', 'min_child_weight':'mn_wght', 
                  'max_delta_step':'mx_stp', 'subsample':'smpl', 'reg_strength':'rg_str', 
                  'num_round':'rnds', 'lambda':'lmbda'}


    # to save the model after each iteration
    feature_string = ''
    for i in range(0,len(feature_list)):
        feature_string = feature_string + feature_list[i] + ';'
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
    pickle.dump([trial_num, train_feature_dir, result], open(model_dir + feature_string + '.model', 'wb'))
    


#sets the global variables, including params passed as args
def set_globals():
    usage = "%prog train_text.json train_labels.csv dev_text.json dev_labels.csv output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='max_iter', default=100,
                      help='Maximum iterations of Bayesian optimization; default=%default')

    (options, args) = parser.parse_args()

    global train_data_filename, train_label_filename, dev_data_filename, dev_label_filename
    global output_dir, train_feature_dir, dev_feature_dir, model_dir, log_filename, trial_num, max_iter
    global num_models
    
    train_data_filename = args[0]
    train_label_filename = args[1]
    dev_data_filename = args[2]
    dev_label_filename = args[3]
    output_dir = args[4]
    num_models = int(args[5])

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
    for i in range(0,3):
        index = priority_q.get()[1]
        print(losses[index])
        print(trials.trials[index]['misc']['vals'])
        print('')
    print('')
        

def main():
    set_globals()
    trials = Trials()
    space = space_manager.get_space(num_models)
    best = fmin(call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=max_iter,
                trials=trials)
    
    print space_eval(space, best)
    printing_best(trials)


if __name__ == '__main__':
    main()





#### CODE GRAVEYARD ####

#        {
#            'model': 'SVM',
#            'regularizer_svm': 'l2',
#            'C_svm': hp.loguniform('C_svm', -1.15, 9.2)
#        },

#    'model': hp.choice('model', [

#        {
#            'model': 'SVM',
#            'regularizer_svm': 'l2',
#            'C_svm': hp.loguniform('C_svm', -1.15, 9.2)
#        },
