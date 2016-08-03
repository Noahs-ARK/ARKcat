from hyperopt import fmin, tpe, hp, Trials, space_eval
import random
import bayesopt, grid_search, random_search

#grid search disregards # models, uses only type of model_types[0]
def get_space(num_models, model_types, search_type, search_space):
    space = {}
    search_module = __import__(search_type)
    for i in range(num_models):
        add_model(str(i), space, model_types, search_module, search_space)
    return space

def add_model(model_num, space, model_types, search_module, search_space):
    set_of_models = []
    for m in model_types:
        if m == 'linear':
            set_of_models.append(search_module.get_linear_model(model_num))
        elif m == 'xgboost':
            set_of_models.append(search_module.get_xgboost_model(model_num))
        elif m == 'cnn':
            set_of_models.append(search_module.get_cnn_model(model_num, search_space))
        else:
            raise NameError('the model ' + m + ' is not implemented.')
    if search_module == bayesopt:
        space['model_' + model_num] = hp.choice('model_' + model_num, set_of_models)
    else:
        space['model_' + model_num] = random.choice(set_of_models)
    # if not are_cnn(model_types):
    space['features_' + model_num] = search_module.get_feats(model_num)

def are_cnn(model_types):
    for model in model_types:
        if model != 'cnn':
            return False
    return True
