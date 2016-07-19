import os
import sys
from optparse import OptionParser
import codecs
import json

import numpy as np
import pandas as pd
from scipy import sparse

from hyperopt import STATUS_OK

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

#import feature_loader

import file_handling as fh

from models_and_data import Data_and_Model_Manager


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
             train_feature_dir, dev_feature_dir, model_dir, word2vec_filename, feats_and_params,
             verbose=1, folds=-1):
    m_and_d = Data_and_Model_Manager(feats_and_params, model_dir, word2vec_filename)
    m_and_d.load_train_and_dev_data(train_data_filename, train_label_filename,
                                    train_feature_dir, dev_data_filename, dev_label_filename,
                                    dev_feature_dir, verbose)
    acc = m_and_d.k_fold_cv(folds)

    print('train acc: ' + str(acc['train_acc']))
    print('dev acc: ' + str(acc['dev_acc']))
    return {'loss': -acc['dev_acc'], 'status': STATUS_OK, 'model': m_and_d}


#DEBUGGING need to remove this stuff
def load_features(data_filename, label_filename, feature_dir, features, verbose, vectorizer=None):
    X_raw, Y = read_data_and_labels(data_filename, label_filename)
    print data_filename
    if vectorizer == None:

        print('size of X: ', X.shape)
        print('size of Y: ', len(Y))
        return X, Y, vectorizer
    else:
        X = vectorizer.transform(X_raw)
        return X, Y


    sys.exit(0)




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
