import cPickle as pickle
from optparse import OptionParser
from classify_test import load_features
from sklearn import metrics
import numpy as np
import Queue
import os
import sys

def predict_one_model(model, data, labels, feature_dir):
    print("\n\n")
    print("In predict_one_model")
    print(model[4])
    print(model[3])
    eval_X, eval_Y = load_features(data, labels, feature_dir, model[4], 1, model[3])
 
    print("predicting labels...")
    print("\n")
    Y_pred = model[0].predict(eval_X)
    return Y_pred, eval_Y

def eval_best_model(models):
    ordered_models = Queue.PriorityQueue()
    test_evals = []
    dev_evals = []
    
    for model in models:
        preds_and_gold = predict_one_model(model, test_data, test_labels, feat_dir)
        acc = metrics.accuracy_score(preds_and_gold[0], preds_and_gold[1])
        test_evals.append(round(acc,5))
        dev_evals.append(round(-model[5]['loss'],5))
        ordered_models.put((model[5]['loss'], -acc, model))

    print("dev accuracy on all models: ")
    print(dev_evals)
    print("test accuracy on all models: ")
    print(test_evals)
    print("best models:")
    best_dev_acc = -1
    

    for i in range(len(models)):
        dev_acc, test_acc, model = ordered_models.get()
        dev_acc = -dev_acc
        test_acc = -test_acc
        if best_dev_acc == -1:
            best_dev_acc = dev_acc
        if dev_acc < best_dev_acc and i > 5:
            break
        
        print('dev acc:  ' + str(dev_acc))
        print('test acc: ' + str(test_acc))
        print('hyperparams: ')
        print(model[1])
        print('')

    
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
    # [model, model_hyperparams, trial_num, train_feature_dir, feature_list, result]
    models = []
    for model_file in os.listdir(model_dir):
        if not model_file.endswith('model'):
            continue
        models.append(pickle.load(open(model_dir + model_file, 'rb')))

    print(len(models))
    print(len(models[0]))
    eval_best_model(models)




if __name__ == '__main__':
    main()



