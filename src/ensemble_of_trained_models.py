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
    dev_ensemble_gold_Y = pred_on_dev_ensemble_data(iter_to_model)
    best_model = None
    best_ensemble = None
    best_model_acc = 0
    best_ens_acc = 0
    print("format: iteration, best model, best ensemble")
    for i in range(2,len(iter_to_model)):
        i = i + 1 #for 1-based indexing
        
        #to get the best singlemodel
        for j in range(i):
            j = j + 1 #stupid 1-based indexing
            if iter_to_model[j]['dev_ensemble_acc'] > best_model_acc:
                best_model_acc = iter_to_model[j]['dev_ensemble_acc']
                best_model = j
            
#        print('best single model so far: ' + str(best_model_acc))
        #find best ensemble of 3        
        best_models = get_best_of_first_i_models(iter_to_model, i)
        #print("dev_ensemble_accuracy of ten best models:")
        #for k in range(len(best_models)):
        #print(best_models[k]['dev_ensemble_acc'])
        #print('')
        
        cur_best_ensemble, cur_best_acc = find_best_ensemble(best_models, dev_ensemble_gold_Y)
        if cur_best_acc > best_ens_acc:
            #print("updating cur best ensemble from " + str(best_ens_acc) + " to " + str(cur_best_acc))
            best_ens_acc = cur_best_acc
            best_ensemble = (best_models[cur_best_ensemble[0]]['model'][2],
                             best_models[cur_best_ensemble[1]]['model'][2],
                             best_models[cur_best_ensemble[2]]['model'][2])
        #print("best ensemble so far:" + str(best_ens_acc))
        print(i, best_model_acc, best_ens_acc)
        
#        print("\n")
    sys.exit(0)

def pred_on_dev_ensemble_data(iter_to_model):
    print("in pred_on_dev_ensemble_data")
    for i in range(len(iter_to_model)):
        i = i + 1
        dev_ensemble_preds, dev_ensemble_gold_Y = predict_one_model(iter_to_model[i]['model'],
                                                                    dev_ensemble_data, dev_ensemble_labels, dev_ensemble_feature_dir)
        
        iter_to_model[i]['dev_ensemble_preds'] = dev_ensemble_preds
        iter_to_model[i]['dev_ensemble_acc'] = ensemble_accuracy([dev_ensemble_preds], dev_ensemble_gold_Y)
    return dev_ensemble_gold_Y
        
    
#makes a map from iteration number to the model. 
def make_iteration_to_model(models):
    iter_to_model = {}

    for i in range(len(models)):
        pred_Y, gold_Y = predict_one_model(models[i], dev_train_data, dev_train_labels, dev_train_feature_dir)
        cur_model_map = {}
        cur_model_map['dev_train_acc'] = ensemble_accuracy([pred_Y], gold_Y)
        cur_model_map['model'] = models[i]
        iter_to_model[models[i][2]] = cur_model_map
    return iter_to_model        


def find_best_ensemble(best_models, dev_ensemble_gold_Y):
    cur_best_ensemble = None
    cur_best_acc = 0
    for i in range(len(best_models)):
        for k in range(i + 1):
            for j in range(k + 1):
                
                ensemble_acc = ensemble_accuracy([best_models[i]['dev_ensemble_preds'], 
                                                  best_models[j]['dev_ensemble_preds'],
                                                  best_models[k]['dev_ensemble_preds']],
                                                 dev_ensemble_gold_Y)
                if ensemble_acc > cur_best_acc:
                    cur_best_ensemble = (i,k,j)
                    cur_best_acc = ensemble_acc
    return cur_best_ensemble, cur_best_acc
                
            
            
def get_best_of_first_i_models(models, i):
    # find top ten models
    ordered_models = Queue.PriorityQueue()
    for j in range(i):
        j = j + 1
        ordered_models.put((-1 * models[j]['dev_train_acc'], j, models[j]))
    ten_best = []
    counter = 0
    while not ordered_models.empty():
        counter = counter + 1
        cur_model = ordered_models.get()
        ten_best.append(cur_model[2])
        if counter == 10:
            break
    return ten_best



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

    
def set_globals(args):
    global model_dir, dev_train_data, dev_train_labels, dev_ensemble_data, dev_ensemble_labels, dev_train_feature_dir, dev_ensemble_feature_dir
    model_dir = args[0]
    dev_train_data = args[1] + 'dev_train.json'
    dev_train_labels = args[1] + 'dev_train.csv'
    dev_ensemble_data = args[1] + 'dev_ensemble.json'
    dev_ensemble_labels = args[1] + 'dev_ensemble.csv'
    dev_train_feature_dir = args[2] + 'dev_train_features'
    dev_ensemble_feature_dir = args[2] + 'dev_ensemble_features'


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
    find_best_ensemble_for_each_iter(models)



if __name__ == '__main__':
    main()



"""
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
"""
