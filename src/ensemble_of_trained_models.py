import cPickle as pickle
from optparse import OptionParser
from classify_test import load_features
from sklearn import metrics
import numpy as np
import Queue
import os
import sys


def find_best_ensemble_for_each_iter(models):
    iter_to_model = make_iteration_to_model(models)
    for i in range(2,len(iter_to_model)):
        i = i + 1 #for 1-based indexing
        best_models = get_best_of_first_i_models(iter_to_model, i)
        dev2_preds = {}
        for k in range(len(best_models)):
            dev2_preds[k], dev2_gold_Y = predict_one_model(best_models[k][1], dev2_data, dev2_labels)
        for k in range(len(best_models)):
            print(dev2_preds[k], best_models[k][0], best_models[k][1][2])
        sys.exit(0)
        find_best_ensemble(best_models)


#makes a map from iteration number to the model. 
def make_iteration_to_model(models):
    iter_to_model = {}

    for i in range(len(models)):
        pred_Y, gold_Y = predict_one_model(models[i], dev1_data, dev1_labels)
        iter_to_model[models[i][2]] = (ensemble_accuracy([pred_Y], gold_Y), models[i])
    return iter_to_model        


def find_best_ensemble(best_models):
    cur_best_ensemble = None
    for i in range(len(best_models)):
        for k in range(i + 1):
            for j in range(k + 1):
                ensemble_acc = ensemble_accuracy()
            
            
def get_best_of_first_i_models(models, i):
    # find top ten models
    ordered_models = Queue.PriorityQueue()
    for j in range(i):
        j = j + 1
        ordered_models.put((-1 * models[j][0], models[j][1]))
    ten_best = []
    counter = 0
    while not ordered_models.empty():
        counter = counter + 1
        ten_best.append(ordered_models.get())
        if counter == 10:
            break
    return ten_best


def predict_with_models(models, args):
    predictions = {}
    gold_Y = None
    for i in range(len(models)):
        predictions[i], gold_Y = predict_one_model(models[i], args)
    
    for i in range(len(predictions)):
        for j in range(i+1,len(predictions)):
            for k in range(j+1,len(predictions)):
                pred_over = sum_pred_overlap([predictions[i], predictions[j], predictions[k]])
                pred_acc = sum_pred_accuracy([predictions[i], predictions[j], predictions[k]], gold_Y)
                ensemble_acc = ensemble_accuracy([predictions[i], predictions[j], predictions[k]], gold_Y)
                print(pred_over, pred_acc, ensemble_acc)
    for i in range(len(predictions)):
        print('accuracy of model ' + str(i) + ': ' + str(sum_pred_accuracy([predictions[i]], gold_Y)))


# finds the number of labels that were predicted the same by the models, then returns the sum
def sum_pred_overlap(preds):
    avg_percent_overlap = 0
    num_comparisons = 0
    for i in range(len(preds)):
        for j in range(len(preds)):
            if i == j:
                continue
            num_comparisons = num_comparisons + 1
            cur_overlap = 0
            for k in range(len(preds[i])):
                if preds[i][k] == preds[j][k]:
                    cur_overlap = cur_overlap + 1
            cur_overlap = float(cur_overlap) / float(len(preds[i]))
            avg_percent_overlap = float(avg_percent_overlap) + float(cur_overlap)
    avg_percent_overlap = float(avg_percent_overlap) / float(num_comparisons)
    return avg_percent_overlap


def sum_pred_accuracy(preds, gold_Y):
    avg_accuracy = 0
    for i in range(len(preds)):
        avg_accuracy = avg_accuracy + metrics.accuracy_score(gold_Y, preds[i])
    avg_accuracy = float(avg_accuracy) / float(len(preds))
    return avg_accuracy


def ensemble_accuracy(preds, gold_Y):
    
    ensemble_preds = []
    
    for i in range(len(preds[0])):
        label_counts = {}
        for j in range(len(preds)):
            cur_label = preds[j][i]
            if cur_label not in label_counts:
                label_counts[cur_label] = 1
            else:
                label_counts[cur_label] = label_counts[cur_label] + 1
        cur_best_label = None
        cur_max_count = 0
        for label in label_counts:
            if label_counts[label] > cur_max_count:
                cur_max_count = label_counts[label]
                cur_best_label = label
        ensemble_preds.append(cur_best_label)
    return metrics.accuracy_score(gold_Y, ensemble_preds)


def predict_one_model(model, data, labels):
    print(model[4])
    print(model[3])
    eval_X, eval_Y = load_features(data, labels, feature_dir, model[4], 1, model[3])
 
    print("\n\npredicting labels...\n")
    Y_pred = model[0].predict(eval_X)
    return Y_pred, eval_Y

    
def set_globals(args):
    global model_dir, dev1_data, dev1_labels, dev2_data, dev2_labels, feature_dir
    model_dir = args[0]
    dev1_data = args[1] + 'dev1.json'
    dev1_labels = args[1] + 'dev1.csv'
    dev2_data = args[1] + 'dev3.json'
    dev2_labels = args[1] + 'dev3.csv'
    feature_dir = args[2]


def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    
    set_globals(args)
    #read in all already-trained models
    print("the params:")
    for model_loc in args:
        print model_loc
    print("")

    #the models were saved in a list like this:
    # [model, model_hyperparams, trial_num, train_feature_dir, feature_list]
    models = []
    for model_file in os.listdir(model_dir):
        if not model_file.endswith('model'):
            continue
        models.append(pickle.load(open(model_dir + model_file, 'rb')))

    print(len(models))
    print(len(models[0]))
#    predict_with_models(models, args)
    find_best_ensemble_for_each_iter(models)



if __name__ == '__main__':
    main()
