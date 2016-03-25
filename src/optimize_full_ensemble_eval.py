import cPickle as pickle
from optparse import OptionParser
from classify_test import load_features
from sklearn import metrics
import numpy as np
import Queue
import os
import sys
from optimize_full_ensemble_classify_test import ensemble_preds

def predict_one_model(model, data, labels, feature_dir):
    print("\n\n")
    print("In predict_one_model")
    # [model, model_hyperparams, trial_num, train_feature_dir, feature_list, result]
    eval_X_1, eval_Y = load_features(data, labels, feature_dir, model['feat_list_1'], 1, model['train_feat_dir'])

    eval_X_2, eval_Y = load_features(data, labels, feature_dir, model['feat_list_2'], 1, model['train_feat_dir'])

    print("predicting labels...")
    print("\n")
 
    eval_Y_prob_1 = model['m1'].predict_proba(eval_X_1)
    eval_Y_prob_2 = model['m2'].predict_proba(eval_X_2)
    eval_Y_pred = ensemble_preds(eval_Y_prob_1, eval_Y_prob_2)

    return eval_Y_pred, eval_Y

def eval_best_model(models):
    best_model = None
    test_evals = []
    dev_evals = []
    for model in models:
        if best_model == None or -best_model['result']['loss'] < -model['result']['loss']:
            best_model = model
        preds_and_gold = predict_one_model(model, test_data, test_labels, feat_dir)
        acc = metrics.accuracy_score(preds_and_gold[0], preds_and_gold[1])
        test_evals.append(round(acc,5))
        dev_evals.append(round(-model['result']['loss'],5))
    print('best accuracy on dev set: ')
    print(-best_model['result']['loss'])
    preds_and_gold = predict_one_model(best_model, test_data, test_labels, feat_dir)
    print("dev accuracy on all models: ")
    print(dev_evals)
    print("test accuracy on all models: ")
    print(test_evals)
    print("test accuracy on model which did best on dev:")
    print(metrics.accuracy_score(preds_and_gold[0], preds_and_gold[1]))
#    print("hyperparam settings for that model:")
#    print(best_model[])


    
def set_globals(args):
    if len(args) < 3:
        print("USEAGE: model_dir data_dir output_dir")
        sys.exit(0)
    global model_dir, test_data, test_labels, feat_dir
    model_dir = args[0]
    test_data = args[1] + 'test.json'
    test_labels = args[1] + 'test.csv'
    feat_dir = args[2] + 'test_features'


def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    
    set_globals(args)
    #print the args
    print("args:")
    for arg in args:
        print arg
    print("")

    #read in all already-trained models
    #the models were saved in a list like this:
    # {'m1': m1, 'hyperparams_1':model_hyperparams_1, 'feat_list_1':feat_list_1, 
    # 'm2': m2, 'hyperparams_2':model_hyperparams_2, 'feat_list_2':feat_list_2, 
    # 'trial_num': trial_num, 'train_feat_dir':train_feature_dir, 'result':result}

    models = []
    for model_file in os.listdir(model_dir):
        if not model_file.endswith('model'):
            continue
        models.append(pickle.load(open(model_dir + model_file, 'rb')))

    print('number of models: ', len(models))
    eval_best_model(models)




if __name__ == '__main__':
    main()



