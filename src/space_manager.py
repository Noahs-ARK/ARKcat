from hyperopt import fmin, tpe, hp, Trials, space_eval
import bayesopt, grid_search, random_search

def get_space(num_models, model_types, search_type):
    space = {}

    search_module = __import__(search_type)
    for i in range(num_models):
        add_model(str(i), space, model_types, search_module)
    return space

def add_model(model_num, space, model_types, search_module):
    set_of_models = []
    for m in model_types:
        if m == 'linear':
            set_of_models.append(search_module.get_linear_model(model_num))
        elif m == 'xgboost':
            set_of_models.append(search_module.get_xgboost_model(model_num))
        elif m == 'cnn':
            set_of_models.append(search_module.get_cnn_model(model_num))
        else:
            raise NameError('the model ' + m + ' is not implemented.')
    #DEBUGGING
    #if cnn then don't add binary option
    if search_module != grid_search:
        space['model_' + model_num] = hp.choice('model_' + model_num, set_of_models)
        space['features_' + model_num] = {
            'nmin_to_max_' + model_num: hp.choice('nmin_to_max_' + model_num,
                                                  [(1,1),(1,2),(1,3),(2,2),(2,3)]),
            'binary_' + model_num: hp.choice('binary_' + model_num, [True, False]),
            'use_idf_' + model_num: hp.choice('transform_' + model_num, [True, False]),
            'st_wrd_' + model_num: hp.choice('st_word_' + model_num, [None, 'english'])
    }
