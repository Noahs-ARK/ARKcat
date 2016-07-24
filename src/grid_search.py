from feature_selector import cnn_feature_selector
import itertools

#returns all combinations from a grid of hyperparameters

#space
def get_cnn_model(model_num):
    feature_selector = cnn_feature_selector()
    general = [feature_selector['delta_'],
                #flex
                # [.15, .3],
                [.15],
                list(feature_selector['filters_']),
                list(feature_selector['kernel_size_']),
                list(feature_selector['kernel_increment_']),
                list(feature_selector['kernel_num_']),
                #dropout
                # [.25, .75],
                [.5],
                list(feature_selector['batch_size_']),
                # iden, relu, and tanh
                feature_selector['activation_fn_'],
                #none, clipped, or penalized
                #learning_rate
                # [.00075, .0015]
                [.001]
                ]
    l2 = [list(feature_selector['l2_'])] + general
    l2_clip = [list(feature_selector['l2_clip_'])] + general
    l2 = list(itertools.product(*l2))
    l2_clip = list(itertools.product(*l2_clip))
    print type(l2)
    l2_models, model_counter = convert_to_dict(l2, 'l2')
    print type(l2_models)
    l2_clip_models, model_counter = convert_to_dict(l2_clip, 'l2_clip', model_counter)
    all_models = l2_models + l2_clip_models
    print 'size of grid %i models' %len(all_models)
    return all_models[int(model_num)]

def convert_to_dict(grid, regularizer, model_counter=0):
    list_of_models = []
    for model in grid:
        model_num = str(model_counter)
        hyperparam_grid = {'model_' + model_num:
                {'model_' + model_num: 'CNN',
                regularizer + '_cnn_' + model_num: model[0],
                'delta_' + model_num: model[1],
                'flex_' + model_num: (True, model[2]),
                'filters_' + model_num: model[3],
                'kernel_size_' + model_num: model[4],
                'kernel_increment_' + model_num: model[5],
                'kernel_num_' + model_num: model[6],
                'dropout_' + model_num: model[7],
                'batch_size_' + model_num: model[8],
                # iden, relu, and tanh
                'activation_fn_' + model_num: model[9],
                'learning_rate_' + model_num: model[10]},
                'features_' + model_num: get_feats(model_num)}
        model_counter += 1
        list_of_models.append(hyperparam_grid)
    return list_of_models, model_counter



"""
    hyperparam_grid = {
            'delta_' + model_num: (feature_selector['delta_']),
            'flex_' + model_num: [(False, 0.0),
                (True, [.15, .3])],
            'filters_' + model_num: list(feature_selector['filters_']),
            'kernel_size_' + model_num: list(feature_selector['kernel_size_']),
            'kernel_increment_' + model_num: list(feature_selector['kernel_increment_']),
            'kernel_num_' + model_num: list(feature_selector['kernel_num_']),
            'dropout_' + model_num: [.25, .75],
            'batch_size_' + model_num: list(feature_selector['batch_size_']),
            # iden, relu, and tanh
            'activation_fn_' + model_num: feature_selector['activation_fn_'],
            #none, clipped, or penalized
            'regularizer_cnn_' + model_num: [
                (None, 0.0),
                ('l2', feature_selector['l2_']),
                ('l2_clip', feature_selector['l2_clip_'])
            ],
            'learning_rate_' + model_num: [.00075, .0015]
    }
    print 'is list:'
    for value in hyperparam_grid.itervalues():
        if type(value) is not list:
            print 'not a list'
"""

def get_feats(model_num):
    return {'nmin_to_max_' + model_num: (1,1),
            'binary_' + model_num: False,
            'use_idf_' + model_num: False,
            'st_wrd_' + model_num: None}
