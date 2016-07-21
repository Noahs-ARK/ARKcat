from feature_selector import cnn_feature_selector

def get_cnn_model(model_num):
    feature_selector = cnn_feature_selector()
    hyperparam_grid = [
            feature_selector['delta_'],
            [#(False, 0.0),
                (True, [.15, .3])],

            feature_selector['filters_'],
            list(feature_selector['kernel_size_']),
            list(feature_selector['kernel_increment_']),
            list(feature_selector['kernel_num_']),
            [.25, .75],
            (feature_selector['batch_size_']),
            # iden, relu, and tanh
            feature_selector['activation_fn_'],
            #none, clipped, or penalized
            [#(None, 0.0),
                ('l2', feature_selector['l2_']),
                ('l2_clip', feature_selector['l2_clip_'])
            ],
            [.00075, .0015]
    ]

    return hyperparam_grid

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

def get_feats(model_num, set_of_models):
    return {
        'model_' + model_num: set_of_models,
        'features_' + model_num: {
            'nmin_to_max_' + model_num: [(1,1),(1,2),(1,3),(2,2),(2,3)],
            'binary_' + model_num: [True, False],
            'use_idf_' + model_num: [True, False],
            'st_wrd_' + model_num: [None, 'english']}
    }
