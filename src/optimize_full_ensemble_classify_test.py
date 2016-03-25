from sklearn.linear_model import LogisticRegression as lr
from classify_test import load_features
import os
from scipy import sparse
import pandas as pd
import feature_loader
import sys
from sklearn import metrics
from hyperopt import STATUS_OK

def classify(train_data_filename_l, train_label_filename_l, dev_data_filename_l, dev_label_filename_l, 
             train_feature_dir_l, dev_feature_dir_l, feat_list_1, args_1, feat_list_2, args_2):

    global train_data_filename, train_label_filename, dev_data_filename, dev_label_filename
    global train_feature_dir, dev_feature_dir
    train_data_filename = train_data_filename_l
    train_label_filename = train_label_filename_l
    dev_data_filename = dev_data_filename_l
    dev_label_filename = dev_label_filename_l 
    train_feature_dir = train_feature_dir_l
    dev_feature_dir = dev_feature_dir_l

    Y_dev_probs_m1, m1, Y_dev = classify_one_model(feat_list_1, **args_1)
    Y_dev_probs_m2, m2, Y_dev = classify_one_model(feat_list_2, **args_2)

    Y_dev_pred = ensemble_preds(Y_dev_probs_m1, Y_dev_probs_m2)
    Y_dev_pred_m1 = ensemble_preds(Y_dev_probs_m1, Y_dev_probs_m1)
    Y_dev_pred_m2 = ensemble_preds(Y_dev_probs_m2, Y_dev_probs_m2)

    dev_acc = metrics.accuracy_score(Y_dev, Y_dev_pred)
    dev_acc_m1 = metrics.accuracy_score(Y_dev, Y_dev_pred_m1)
    dev_acc_m2 = metrics.accuracy_score(Y_dev, Y_dev_pred_m2)
    
    print('dev acc: ' + str(dev_acc))
    print('dev acc m1: ' + str(dev_acc_m1))
    print('dev acc m2: ' + str(dev_acc_m2))
    neg_loss = dev_acc

    return {'loss': -neg_loss, 'status': STATUS_OK, 'm1': m1, 'm2': m2}

def ensemble_preds(m1_probs, m2_probs):
    preds = []
    for i in range(len(m1_probs)):
        prob_of_zero = (m1_probs[i][0] + m2_probs[i][0])/2
        prob_of_one = (m1_probs[i][1] + m2_probs[i][1])/2
        if prob_of_zero > prob_of_one:
            preds.append(0)
        else:
            preds.append(1)
    return preds
                

def classify_one_model(feature_list, model_type='LR', regularizer='l1', alpha=1.0, converg_tol=0.01, verbose=1, folds=2, n_jobs=-1, score_eval='f1'):

    if model_type == 'LR':
        model = lr(penalty=regularizer, C=alpha, tol=converg_tol)
    elif model_type == 'SVM':
        model = svm.LinearSVC(penalty=regularizer, C=alpha, tol=converg_tol)
    else:
        sys.exit('Model type ' + model_type + ' not supported')

    train_X, train_Y = load_features(train_data_filename, train_label_filename, train_feature_dir, 
                                     feature_list, verbose)
    # Try loading dev data using train vocabulary, and not saving dev feature extractions
    dev_X, dev_Y = load_features(dev_data_filename, dev_label_filename, dev_feature_dir,
                                     feature_list, verbose, vocab_source=train_feature_dir)

    model.fit(train_X, train_Y)
    dev_pred_prob_Y = model.predict_proba(dev_X)
    
    return dev_pred_prob_Y, model, dev_Y
