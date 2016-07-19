import cPickle as pickle
from optparse import OptionParser
from classify_test import load_features
from sklearn import metrics
import numpy as np
import Queue
import os
import sys

def eval_best_model_of_iter_i(models, max_iter, print_k_models_per_iter):
    ordered_models = Queue.PriorityQueue()
    test_evals = []
    dev_evals = []

    model_counter = 0
    for model in models:
        if model[0] > max_iter:
            continue
        sys.stdout.write('evaluating model ' + str(model_counter) + '...')
        acc = model[2]['model'].predict_acc_from_file(test_data, test_labels)
        test_evals.append(round(acc,5))
        dev_evals.append(round(-model[2]['loss'],5))
        ordered_models.put((model[2]['loss'], -acc, model[2]))
        sys.stdout.write('done!\n')
        model_counter = model_counter + 1


    best_dev_acc = -1


    for i in range(len(models)):
        dev_acc, test_acc, model = ordered_models.get()
        dev_acc = -dev_acc
        test_acc = -test_acc
        if best_dev_acc == -1:
            best_dev_acc = dev_acc
        if dev_acc < best_dev_acc and i > print_k_models_per_iter:
            break

        print('test acc: ' + str(test_acc))
        print('dev acc:  ' + str(dev_acc))
        print('hyperparams: ')
        print(model['model'].feats_and_params)
        print('')

def eval_best_model(models):
    print_k_models_per_iter = 1
    for i in range(len(models)):
        i = i + 1
        print("best models for iteration " + str(i) + ":")
        eval_best_model_of_iter_i(models, i, print_k_models_per_iter)
        print("")
        print("")


def set_globals(args):
    if len(args) < 4:
        print("USEAGE: model_dir data_dir word2vec_filename output_dir")
        sys.exit(0)
    global model_dir, test_data, test_labels, feat_dir
    model_dir = args[0]
    test_data = args[1] + 'test.data'
    test_labels = args[1] + 'test.labels'
    word2vec_filename = args[2]
    feat_dir = args[3] + 'test_features'


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
    #[-dev_acc, status, Data_and_Model_Manager]
    models = []
    sys.stdout.write('loading the trained models...\n')
    model_counter = 0
    for model_file in os.listdir(model_dir):
        if not model_file.endswith('model'):
            continue
        models.append(pickle.load(open(model_dir + model_file, 'rb')))
        print('loaded model ' + str(model_counter))
        model_counter = model_counter + 1

    eval_best_model(models)




if __name__ == '__main__':
    main()
