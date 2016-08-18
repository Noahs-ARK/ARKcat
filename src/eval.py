import cPickle as pickle
from optparse import OptionParser
from classify_test import load_features
from sklearn import metrics
import numpy as np
import Queue
import os
import sys

def eval_k_best_models(models, print_k_models_per_iter):

    test_evals = []
    dev_evals = []
    models_and_acc = {}
    model_counter = 0
    for model in models:
        sys.stdout.write('evaluating model ' + str(model_counter) + '...')
        sys.stdout.flush()
        acc = model[2]['model'].predict_acc_from_file(test_data, test_labels)
        test_evals.append(round(acc,5))
        dev_evals.append(round(-model[2]['loss'],5))
        #DEBUGGING should sort by dev acc, then sample among ties.
        models_and_acc[model[0]] = (model[2]['loss'], model[0], acc * -1, model)
        sys.stdout.write('done! with accuracy ' + str(round(acc,5)) + '\n')
        model_counter = model_counter + 1
    sys.stdout.flush()
    print('best ' + str(print_k_models_per_iter) + ' models per iter:')

    for i in range(len(models)):
        ordered_models = Queue.PriorityQueue()
        for j in range(i + 1):
            ordered_models.put(models_and_acc[j + 1])
        print('')
        print('best models for iter ' + str(i + 1))
        for j in range(print_k_models_per_iter):
            dev_acc, iteration, test_acc, model = ordered_models.get()
            print('test acc: ' + str(test_acc))
            print('dev acc:  ' + str(dev_acc))
            print('hyperparams: ')
            print(model[2]['model'].feats_and_params)
            print('')
        print('')
        sys.stdout.flush()

def set_globals(args):
    if len(args) < 3:
        print("USEAGE: model_dir data_dir output_dir")
        sys.exit(0)
    global model_dir, test_data, test_labels, feat_dir
    model_dir = args[0]
    test_data = args[1] + 'test.data'
    test_labels = args[1] + 'test.labels'
    feat_dir = args[2] + 'test_features'

def find_models_in_dir(directory, model_counter=0, models=[]):
    for model_file in os.listdir(directory):
        if not model_file.endswith('model'):
            continue
        models.append(pickle.load(open(directory + model_file, 'rb')))
        print('loaded model ' + str(model_counter))
        sys.stdout.flush()
        model_counter = model_counter + 1
    return models, model_counter


def main():
    print "got to eval"
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
    sys.stdout.write('loading the trained models...\n')
    models, model_counter = find_models_in_dir(model_dir)
    if not models:
        for directory in os.listdir(model_dir):
            models, model_counter = find_models_in_dir(model_dir + directory + '/', model_counter, models)
    print_k_models_per_iter = 1
    print models
    eval_k_best_models(models, print_k_models_per_iter)

if __name__ == '__main__':
    main()
