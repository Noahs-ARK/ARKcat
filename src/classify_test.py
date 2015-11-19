import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse

from hyperopt import STATUS_OK

from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score

import feature_loader

import file_handling as fh

def main():
    """ test function """
    # Handle input options and arguments
    usage = "%prog text.json labels.csv feature_dir feature1 [feature2...]"
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

    data_filename = args[0]
    label_filename = args[1]
    feature_dir = args[2]
    features = args[3:]

    classify(data_filename, label_filename, feature_dir, features, model_type=model_type,
             regularizer=regularizer, alpha=alpha, verbose=verbose)


def classify(data_filename, label_filename, feature_dir, list_of_features, model_type='LR',
             regularizer='l1', alpha=1.0, verbose=1,folds=2,n_jobs=-1,score_eval='f1'):

    labels = pd.read_csv(label_filename, header=0, index_col=0)
    items_to_load = labels.index

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    # for each feature in feature_list:
    items = None
    feature_matrices = []
    column_names = []
    print "Loading features"
    for feature in list_of_features:
        feature_description = feature
        rows, columns, counts = feature_loader.load_feature(feature_description, feature_dir, data_filename,
                                                            items_to_load, verbose=1)
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
    column_names = np.concatenate(column_names)
    if verbose > 0:
        print "Full feature martix size:", X.shape

    #return items, column_names, X
    if model_type == 'LR':
        model = lr(penalty=regularizer, C=alpha)
    elif model_type == 'SVM':
        model = svm.LinearSVC(C=alpha, penalty=regularizer)
    else:
        sys.exit('Model type ' + model_type + ' not supported')

    y = labels.as_matrix().ravel()

    skf = StratifiedKFold(y, folds,random_state=17)
    f1 = cross_val_score(model, X, y, cv=skf,scoring=score_eval,n_jobs=n_jobs).mean()

    print f1
    return {'loss': -f1, 'status': STATUS_OK, 'model': model}





if __name__ == '__main__':
    main()
