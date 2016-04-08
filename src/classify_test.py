import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse

from hyperopt import STATUS_OK

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

import feature_loader

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
             train_feature_dir, dev_feature_dir, feats_and_params, verbose=1, folds=-1):
    m_and_d = Data_and_Model_Manager(feats_and_params)
    train_acc = m_and_d.train_models(train_data_filename, train_label_filename, 
                                     train_feature_dir, verbose)
    dev_acc = m_and_d.predict_acc(dev_data_filename, dev_label_filename, dev_feature_dir, verbose)
    
    print('train acc: ' + str(train_acc))
    print('dev acc: ' + str(dev_acc))
    return {'loss': -dev_acc, 'status': STATUS_OK, 'model': m_and_d}
    
          

#    model = Model(set_of_params)

#    train_X, train_Y = load_features(train_data_filename, train_label_filename, train_feature_dir, 
#                                     feature_list, verbose)
    #if we have separate dev data, so we don't need cross validation
#    if folds < 1:
        # Try loading dev data using train vocabulary, and not saving dev feature extractions
#        dev_X, dev_Y = load_features(dev_data_filename, dev_label_filename, dev_feature_dir,
#                                     feature_list, verbose, vocab_source=train_feature_dir)

#        model.train(train_X, train_Y)
#        dev_Y_pred = model.predict(dev_X)
#        train_Y_pred = model.predict(train_X)
        
#        dev_f1, dev_acc, train_f1, train_acc = metrics.f1_score(dev_Y, dev_Y_pred), metrics.accuracy_score(dev_Y, dev_Y_pred), metrics.f1_score(train_Y, train_Y_pred), metrics.accuracy_score(train_Y, train_Y_pred)
#        print('train acc: ' + str(train_acc))
#        print('dev acc: ' + str(dev_acc))
#        neg_loss = dev_acc
    #if we don't have separate dev data, so we need cross validation
#    else:
#        skf = StratifiedKFold(train_Y, folds,random_state=17)
#        neg_loss = cross_val_score(model, train_X, train_Y, cv=skf,scoring=score_eval,n_jobs=n_jobs).mean()
#        print('crossvalidation f1: ' + str(f1))

#    return {'loss': -neg_loss, 'status': STATUS_OK, 'model': model}



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
