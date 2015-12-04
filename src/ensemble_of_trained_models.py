import cPickle as pickle
from optparse import OptionParser
from classify_test import load_features
from sklearn import metrics
import numpy as np

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

def predict_one_model(model, args):
    eval_X, eval_Y = load_features(args[-3], args[-2], args[-1], model[4], 1, model[3])

    print("\n\npredicting labels...\n")
    Y_pred = model[0].predict(eval_X)
#    print(len(Y_pred))
#    print(Y_pred)
    return Y_pred, eval_Y
    


def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()
    
    #read in all already-trained models
    print("the params:")
    for model_loc in args:
        print model_loc
    print("")
    #parser.add_option('-m', dest='max_iter', default=4,
    #                  help='Maximum iterations of Bayesian optimization; default=%default')
    
    #the models were saved in a list like this:
    # [model, model_hyperparams, num_trials]
    models = []
    for i in range(len(args)-3):
        model_loc = args[i]
        models.append(pickle.load(open(model_loc, 'rb')))
        models[i][1] = pickle
    print(len(models))
    print(len(models[0]))
#    print(len(models[0][1]))
    print(models[0][1])
    predict_with_models(models, args)




if __name__ == '__main__':
    main()
