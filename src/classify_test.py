import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse

from hyperopt import STATUS_OK

from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

import feature_loader

import file_handling as fh

def main():
    """ test function """
    # Handle input options and arguments
    usage = "%prog train_text.json train_labels.csv dev_text.json dev_labels.csv train_feature_dir dev_feature_dir feature1 [feature2...]"
    parser = OptionParser(usage=usage)
    parser.add_option('-m', dest='model', default='LR',
                      help='Model type [LR|SVM]; default=%default')
    parser.add_option('-r', dest='regularizer', default='l1',
                      help='Regularizer [l1|l2]; default=%default')
    parser.add_option('-a', dest='alpha', default=1,
                      help='regularization strength; default=%default')
    parser.add_option('-v', dest='verbose', default=1,
                      help='Verbosity levels; default=%default')

    (options, args) = parser.parse_args()
    verbose = options.verbose
    model_type = options.model
    regularizer = options.regularizer
    alpha = float(options.alpha)

    train_data_filename = args[0]
    train_label_filename = args[1]
    dev_data_filename = args[2]
    dev_label_filename = args[3]

    train_feature_dir = args[4]
    dev_feature_dir = args[5]
    features = args[6:]

    classify(train_data_filename, train_label_filename, dev_data_filename, dev_label_filename,
             train_feature_dir, dev_feature_dir, features, model_type=model_type,
             regularizer=regularizer, alpha=alpha, verbose=verbose, folds=0)


def classify(train_data_filename, train_label_filename, dev_data_filename, dev_label_filename, 
             train_feature_dir, dev_feature_dir, feature_list, model_type='LR', 
             regularizer='l1', alpha=1.0, converg_tol=0.01, verbose=1, folds=2, n_jobs=-1, score_eval='f1'):
    
    if model_type == 'LR':
        model = lr(penalty=regularizer, C=alpha, tol=converg_tol)
    elif model_type == 'SVM':
        model = svm.LinearSVC(penalty=regularizer, C=alpha, tol=converg_tol)
    else:
        sys.exit('Model type ' + model_type + ' not supported')

    train_X, train_Y = load_features(train_data_filename, train_label_filename, train_feature_dir, 
                                     feature_list, verbose)
    #if we have separate dev data, so we don't need cross validation
    if folds < 1:
        #dev_X, dev_Y = load_features(dev_data_filename, dev_label_filename, dev_feature_dir,
        #                             feature_list, verbose)

        # Try loading dev data using train vocabulary, and not saving dev feature extractions
        dev_X, dev_Y = load_features(dev_data_filename, dev_label_filename, dev_feature_dir,
                                     feature_list, verbose, vocab_source=train_feature_dir)
#        print("size of train_X[0]: ", train_X[0].shape)
#        for thing in train_X[0]:
#            print(thing)
#        for thing in train_Y:
#            print(thing),
#            if not (thing == 1 or thing == 0):
#                print('found something bad')
#        import pdb; pdb.set_trace() #DEBUGGING
        f1 = no_cross_validation(train_X, train_Y, dev_X, dev_Y, model)
        print('dev f1: ' + str(f1))
    #if we don't have separate dev data, so we need cross validation
    else:
        skf = StratifiedKFold(train_Y, folds,random_state=17)
        f1 = cross_val_score(model, train_X, train_Y, cv=skf,scoring=score_eval,n_jobs=n_jobs).mean()
        print('crossvalidation f1: ' + str(f1))

    return {'loss': -f1, 'status': STATUS_OK, 'model': model}

def no_cross_validation(X_train, Y_train, X_dev, Y_dev, model):
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    Y_dev_pred = model.predict(X_dev)

    train_f1 = metrics.accuracy_score(Y_train, Y_train_pred)
    dev_f1 = metrics.accuracy_score(Y_dev, Y_dev_pred)

    return dev_f1


def load_features(data_filename, label_filename, feature_dir, feature_list, verbose, vocab_source=None):
    
    labels = pd.read_csv(label_filename, header=0, index_col=0)
    items_to_load = labels.index

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    # for each feature in feature_list:
    items = None
    feature_matrices = []
    column_names = []
    print "Loading features"
    for feature in feature_list:
        feature_description = feature
        print("DEBUGGING:")
        print(feature_description)
        print(feature_dir)
        print(data_filename)
        print(items_to_load)
        print(vocab_source)
        print('\n')

        rows, columns, counts = feature_loader.load_feature(feature_description, feature_dir, 
                                data_filename, items_to_load, verbose=1, vocab_source=vocab_source)
        if items is None:
            items = rows
        else:
            assert items.tolist() == rows.tolist()
        if verbose > 0:
            print "Loaded", feature, "with shape", counts.shape
        feature_matrices.append(counts)
        column_names.append(columns)

    # concatenate all features together
    X = sparse.csr_matrix(sparse.hstack(feature_matrices))

    if verbose > 0:
        print "Full feature martix size:", X.shape
    Y = labels.as_matrix().ravel()

    return X, Y



if __name__ == '__main__':
    main()
