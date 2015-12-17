import os
import codecs
import datetime
import cPickle as pickle
from optparse import OptionParser

from hyperopt import fmin, tpe, hp, Trials, space_eval

import classify_test


data_filename = None
label_filename = None
feature_dir = None
output_dir = None
log_filename = None

space = {
    'model': hp.choice('model', [
        {
            'model': 'SVM',
            'regularizer_svm': 'l2',
            'C_svm': hp.loguniform('C_svm', -1.15, 9.2)
        },
        {
            'model': 'LR',
            'regularizer_lr': hp.choice('regularizer_lr', ['l1', 'l2']),
            'alpha_lr': hp.loguniform('alpha_lr', -1.15, 9.2)
        }
    ]),
    'features': {
        'unigrams':
            {
                'transform': hp.choice('u_transform', ['None', 'binarize', 'tfidf']),
                'min_df': hp.choice('u_min_df', [1,2,3,4,5])
            },

        'bigrams':
            hp.choice('bigrams', [
                {
                    'use': False
                },
                {
                    'use': True,
                    'transform': hp.choice('b_transform', ['None', 'binarize', 'tfidf']),
                    'min_df': hp.choice('b_min_df',[1,2,3,4,5])
                }
            ]),
        }
    }

def call_experiment(args):
    global trial_num
    trial_num = trial_num + 1
    feature_list, description, kwargs = wrangle_params(args)
    result = classify_test.classify(train_data_filename, train_label_filename, dev_data_filename, 
                                    dev_label_filename, train_feature_dir, dev_feature_dir, 
                                    feature_list, **kwargs)
    with codecs.open(log_filename, 'a') as output_file:
        output_file.write(str(datetime.datetime.now()) + '\t' + ' '.join(feature_list) + '\t' + ' '.join(description) +
                          '\t' + str(-result['loss']) + '\n')
    save_model(result['model'], feature_list, kwargs)

    print("\nFinished iteration " + str(trial_num) + ".\n\n\n")
    return result

def wrangle_params(args):
    kwargs = {}

    # WARNING THIS IS A HACK! Should pass this is as a param
    kwargs['folds'] = 0

    print('')
    print('the args:')
    print(args)
    model = args['model']['model']
    kwargs['model_type'] = model
    if model == 'SVM':
        kwargs['regularizer'] = args['model']['regularizer_svm']
        kwargs['alpha'] = args['model']['C_svm']
    elif model == 'LR':
        kwargs['regularizer'] = args['model']['regularizer_lr']
        kwargs['alpha'] = args['model']['alpha_lr']

    feature_list = []
    unigrams = 'ngrams,n=1' + \
               ',transform=' + args['features']['unigrams']['transform'] + \
               ',min_df=' + str(args['features']['unigrams']['min_df'])
    feature_list.append(unigrams)
    if args['features']['bigrams']['use']:
        bigrams = 'ngrams,n=2' + \
                  ',transform=' + args['features']['bigrams']['transform'] + \
                  ',min_df=' + str(args['features']['bigrams']['min_df'])
        feature_list.append(bigrams)

    print feature_list
    description = [str(k) + '=' + str(v) for (k, v) in kwargs.items()]
    return feature_list, description, kwargs
    

def save_model(model, feature_list, model_hyperparams):
    # to save the model after each iteration
    feature_string = ''
    for i in range(0,len(feature_list)):
        feature_string = feature_string + feature_list[i] + ';'
    for hparam in model_hyperparams:
        feature_string = feature_string + hparam + '=' + str(model_hyperparams[hparam]) + ';'
    feature_string = feature_string[:-1]
    pickle.dump([model, model_hyperparams, trial_num, train_feature_dir, feature_list], open(model_dir + feature_string + '.model', 'wb'))
    


def main():
    usage = "%prog train_text.json train_labels.csv dev_text.json dev_labels.csv output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='max_iter', default=10,
                      help='Maximum iterations of Bayesian optimization; default=%default')

    (options, args) = parser.parse_args()
    max_iter = int(options.max_iter)

    global train_data_filename, train_label_filename, dev_data_filename, dev_label_filename
    global output_dir, train_feature_dir, dev_feature_dir, model_dir, log_filename, trial_num

    train_data_filename = args[0]
    train_label_filename = args[1]
    dev_data_filename = args[2]
    dev_label_filename = args[3]
    output_dir = args[4]

    train_feature_dir = output_dir + '/train_features/'
    dev_feature_dir = output_dir + '/dev_features/'
    model_dir = output_dir + '/saved_models/'
    
    trial_num = 0
    
    for directory in [output_dir, train_feature_dir, dev_feature_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    log_filename = os.path.join(output_dir, 'log.txt')

    with open(log_filename, 'w') as logfile:
        logfile.write(','.join([train_data_filename, train_label_filename, dev_data_filename, 
                                dev_label_filename, train_feature_dir, dev_feature_dir, output_dir]) + '\n')

    trials = Trials()
    best = fmin(call_experiment,
                space=space,
                algo=tpe.suggest,
                max_evals=max_iter,
                trials=trials)
    
    print space_eval(space, best)
    print "losses:", [-l for l in trials.losses()]
    print("number of trials: " + str(len(trials.trials)))


if __name__ == '__main__':
    main()
