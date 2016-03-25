import os, sys
import codecs
import datetime
import cPickle as pickle
from optparse import OptionParser
import numpy as np

from hyperopt import fmin, tpe, hp, Trials, space_eval

import optimize_full_ensemble_classify_test


data_filename = None
label_filename = None
feature_dir = None
output_dir = None
log_filename = None

space = {

    'model_1':
        {
            'model_1': 'LR',
            'regularizer_lr_1': hp.choice('regularizer_lr_1',
                [
                    ('l1', hp.uniform('l1_strength_1', 0,1)),
                    ('l2', hp.uniform('l2_strength_1', 0,1))

#                    ('l1', hp.loguniform('l1_strength', np.log(1e-7), np.log(10**2))),
#                    ('l2', hp.loguniform('l2_strength', np.log(1e-7), np.log(100)))
                ]),
            'converg_tol_1': hp.loguniform('converg_to_l', -10, -1)

        },
    'features_1': {
        'unigrams_1':
            {
                'transform_1': hp.choice('u_transform_1', ['None', 'binarize', 'tfidf']),
                'min_df_1': hp.choice('u_min_df_1',[1,2,3,4,5])
            },
        'bigrams_1':
            hp.choice('bigrams_1', [
                {
                    'use_1': False
                },
                {
                    'use_1': True,
                    'transform_1': hp.choice('b_transform_1', ['None', 'binarize', 'tfidf']),
                    'min_df_1': hp.choice('b_min_df_1',[1,2,3,4,5])
                }
            ]),
    },
    'model_2':
        {
            'model_2': 'LR',
            'regularizer_lr_2': hp.choice('regularizer_lr_2',
                [
                    ('l1', hp.uniform('l1_strength_2', 0,1)),
                    ('l2', hp.uniform('l2_strength_2', 0,1))

#                    ('l1', hp.loguniform('l1_strength', np.log(1e-7), np.log(10**2))),
#                    ('l2', hp.loguniform('l2_strength', np.log(1e-7), np.log(100)))
                ]),
            'converg_tol_2': hp.loguniform('converg_tol_2', -10, -1)

        },
    'features_2': {
        'unigrams_2':
            {
                'transform_2': hp.choice('u_transform_2', ['None', 'binarize', 'tfidf']),
                'min_df_2': hp.choice('u_min_df_2',[1,2,3,4,5])
            },
        'bigrams_2':
            hp.choice('bigrams_2', [
                {
                    'use_2': False
                },
                {
                    'use_2': True,
                    'transform_2': hp.choice('b_transform_2', ['None', 'binarize', 'tfidf']),
                    'min_df_2': hp.choice('b_min_df_2',[1,2,3,4,5])
                }
            ]),
    }}




def call_experiment(args):
    global trial_num
    trial_num = trial_num + 1
    feats_args_desc = wrangle_params(args)
    
    result = optimize_full_ensemble_classify_test.classify(train_data_filename, 
                 train_label_filename, dev_data_filename, dev_label_filename, 
                 train_feature_dir, dev_feature_dir, feats_args_desc['1'][0], 
                                                           feats_args_desc['1'][2], feats_args_desc['2'][0], feats_args_desc['2'][2])
    with codecs.open(log_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + ' '.join(feats_args_desc['1'][0]) + '\t' + ' '.join(feats_args_desc['1'][1]) + '\t' + ' '.join(feats_args_desc['2'][0]) +  '\t' + ' '.join(feats_args_desc['2'][1]) + 
                          '\t' + str(-result['loss']) + '\n')
    save_model(result['m1'], result['m2'], feats_args_desc['1'][0], feats_args_desc['2'][0], feats_args_desc['1'][2], feats_args_desc['2'][2], result)

    print("\nFinished iteration " + str(trial_num) + ".\n\n\n")
    return result

def wrangle_params(args):
    print('')
    print('the args:')
    print(args)

    
    feature_list_1, description_1, kwargs_1 = wrangle_two_model_params(args, '_1')
    feature_list_2, description_2, kwargs_2 = wrangle_two_model_params(args, '_2')
    
    return {'1': [feature_list_1, description_1, kwargs_1], 
            '2': [feature_list_2, description_2, kwargs_2]}

def wrangle_two_model_params(args, model_no):
    kwargs = {}
    kwargs['folds'] = 0
    kwargs['model_type'] = args['model' + model_no]['model' + model_no]
    kwargs['regularizer'] = args['model' + model_no]['regularizer_lr' + model_no][0]
    kwargs['alpha'] = args['model' + model_no]['regularizer_lr' + model_no][1]
    kwargs['converg_tol'] = args['model' + model_no]['converg_tol' + model_no]

    feature_list = []
    unigrams = 'ngrams,n=1' + \
               ',transform=' + args['features' + model_no]['unigrams' + model_no]['transform' + model_no] + \
               ',min_df=' + str(args['features' + model_no]['unigrams' + model_no]['min_df' + model_no])
    feature_list.append(unigrams)
    if args['features' + model_no]['bigrams' + model_no]['use' + model_no]:
        bigrams = 'ngrams,n=2' + \
                  ',transform=' + args['features' + model_no]['bigrams' + model_no]['transform' + model_no] + \
                  ',min_df=' + str(args['features' + model_no]['bigrams' + model_no]['min_df' + model_no])
        feature_list.append(bigrams)

    description = [str(k) + '=' + str(v) for (k, v) in kwargs.items()]
    return feature_list, description, kwargs
    

def save_model(m1, m2, feat_list_1, feat_list_2, model_hyperparams_1, model_hyperparams_2, result):
    # to save the model after each iteration
    feature_string_1 = ''
    for i in range(0,len(feat_list_1)):
        feature_string_1 = feature_string_1 + feat_list_1[i] + ';'
    for hparam in model_hyperparams_1:
        feature_string_1 = feature_string_1 + hparam + '=' + str(model_hyperparams_1[hparam]) + ';'
    feature_string_1 = feature_string_1[:-1]

    feature_string_2 = ''
    for i in range(0,len(feat_list_2)):
        feature_string_2 = feature_string_2 + feat_list_2[i] + ';'
    for hparam in model_hyperparams_2:
        feature_string_2 = feature_string_2 + hparam + '=' + str(model_hyperparams_2[hparam]) + ';'
    feature_string_2 = feature_string_2[:-1]

    stuff_to_save = {'m1': m1, 'hyperparams_1':model_hyperparams_1, 'feat_list_1':feat_list_1, 
     'm2': m2, 'hyperparams_2':model_hyperparams_2, 'feat_list_2':feat_list_2, 
     'trial_num': trial_num, 'train_feat_dir':train_feature_dir, 'result':result}

#    pickle.dump(stuff_to_save, open(model_dir + '{' + feature_string_1 + '},{' + feature_string_2
#                                    + '}.model', 'wb'))
    
    pickle.dump(stuff_to_save, open(model_dir + '{' + feature_string_1 + '}.model', 'wb'))


#sets the global variables, including params passed as args
def set_globals():
    usage = "%prog train_text.json train_labels.csv dev_text.json dev_labels.csv output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='max_iter', default=100,
                      help='Maximum iterations of Bayesian optimization; default=%default')

    (options, args) = parser.parse_args()

    global train_data_filename, train_label_filename, dev_data_filename, dev_label_filename
    global output_dir, train_feature_dir, dev_feature_dir, model_dir, log_filename, trial_num, max_iter
    
    train_data_filename = args[0]
    train_label_filename = args[1]
    dev_data_filename = args[2]
    dev_label_filename = args[3]
    output_dir = args[4]

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


def main():
    set_globals()
    trials = Trials()
    best = fmin(call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=max_iter,
                trials=trials)
    
    print space_eval(space, best)
    print "losses:", [-l for l in trials.losses()]
    print('the best loss: ', max([-l for l in trials.losses()]))
    print("number of trials: " + str(len(trials.trials)))


if __name__ == '__main__':
    main()
