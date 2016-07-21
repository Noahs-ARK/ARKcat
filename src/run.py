import os, sys
import codecs
import datetime
import cPickle as pickle
import Queue as queue

from optparse import OptionParser
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, space_eval
from sklearn.grid_search import GridSearchCV#, RandomizedSearchCV

import classify_test
import space_manager


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
                                    model_dir, word2vec_filename, feats_and_args, folds=num_folds)



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
    elif model == 'CNN':
        print args['model_' + model_num]['regularizer_cnn_' + model_num][0]
        print args['model_' + model_num]['regularizer_cnn_' + model_num][1]
        kwargs['word_vector_init'] = args['model_' + model_num]['word_vectors_' + model_num][0]
        kwargs['word_vector_update'] = args['model_' + model_num]['word_vectors_' + model_num][1]
        kwargs['delta'] = args['model_' + model_num]['delta_' + model_num]
        kwargs['flex'] = (args['model_' + model_num]['flex_' + model_num])[0]
        kwargs['flex_amt'] = (args['model_' + model_num]['flex_' + model_num])[1]
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

    #note to Jesse: code in
    #passing English will break CNN because some examples become null
    if model == 'CNN':
        features['stop_words'] = None
        features['binary'] = False

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
def set_globals():
    usage = "%prog train_text.json train_labels.csv dev_text.json dev_labels.csv output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='max_iter', default=30,
                      help='Maximum iterations of Bayesian optimization; default=%default')

    (options, args) = parser.parse_args()

    global train_data_filename, train_label_filename, dev_data_filename, dev_label_filename
    global output_dir, train_feature_dir, dev_feature_dir, model_dir, word2vec_filename, log_filename
    global trial_num, max_iter, num_models, model_types, search_type, num_folds

    train_data_filename = args[0] + 'train.data'
    train_label_filename = args[0] + 'train.labels'
    dev_data_filename = args[0] + 'dev.data'
    dev_label_filename = args[0] + 'dev.labels'
    word2vec_filename = args[1]
    output_dir = args[2]
    num_models = int(args[3])
    model_types = args[4].split('-')
    search_type = args[5]
    num_folds = int(args[6])
    print('train data filename: ',train_data_filename)

    train_feature_dir = output_dir + '/train_features/'
    dev_feature_dir = output_dir + '/dev_train_features/'
    model_dir = output_dir + '/saved_models/'

    trial_num = 0
    max_iter = int(options.max_iter)

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


def main():
    print("Made it to the start of main!")
    set_globals()
    trials = Trials()
    space = space_manager.get_space(num_models, model_types, search_type)
    if search_type == 'grid_search':
        best = run_grid_search(space, model_types)
    else:
        best = fmin(call_experiment,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_iter,
                    trials=trials)
    # #need a classify_test method
    # elif search_type == 'grid_search':
    #     best = run_grid_search(space, model_types)
    # else: #search_type == 'random_search'
    #     best = run_random_search(space, model_types, max_iter)

    print space_eval(space, best)
    printing_best(trials)

def run_grid_search(space, model_types):
    grid_search = GridSearchCV(model_types, param_grid=space)
    grid_search.fit(X, y)
    report(grid_search.grid_scores_)

# def run_random_search(space, model_types, n_iter):
#     random_search = RandomizedSearchCV(model_types, param_distributions=space,
#                                        n_iter=n_iter)
#     random_search.fit(X, y)
#     return best(random_search.grid_scores_)

# Utility function to report best scores
def best(grid_scores, n_top=1):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


if __name__ == '__main__':
    main()
