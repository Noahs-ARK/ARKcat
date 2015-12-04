import cPickle as pickle
from optparse import OptionParser
from classify_test import load_features


def predict_with_models(models, args):
    predict_one_model(models[0], args)

def predict_one_model(model, args):
    eval_X, eval_Y = load_features(args[-3], args[-2], args[-1], model[4], 1, model[3])

    print("\n\npredicting labels...\n")
    Y_pred = model[0].predict(eval_X)
    print(len(Y_pred))
    print(Y_pred)
    


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
