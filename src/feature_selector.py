def cnn_feature_selector(search_space):
    if search_space == 'reg':
        return {'model_': 'CNN',
                'delta_': [False],
                'flex_amt_': (0.15),
                'filters_': (100),
                'kernel_size_': (3),
                'kernel_increment_': (1),
                'kernel_num_': (3),
                'dropout_': (0, 0.75),
                'batch_size_': (50),
                'activation_fn_': ['relu'],
                'l2_': (-5.0, -1.0),
                'l2_clip_': (1.0, 5.0),
                'no_reg': True,
                'search_lr': False,
                'grid': [1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 4, 3]
        }
    elif search_space == 'arch':
        return {'model_': 'CNN',
                'delta_': [True, False],
                'flex_amt_': (0.0, 0.3),
                'filters_': (100, 600),
                'kernel_size_': (2, 15),
                'kernel_increment_': (0,5),
                'kernel_num_': (1,5),
                'dropout_': (0.25, 0.75),
                'batch_size_': (10, 200),
                'activation_fn_': ['iden', 'relu', 'elu'],
                'l2_': (-8.0,-2.0),
                'l2_clip_': (2.0,10.0),
<<<<<<< HEAD
                #try no regularization
                'no_reg': False,
                #search learning rates (False automatically use default)
                'search_lr': False,
                'grid': [1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2]

=======
                'l2_extras': [],
                'clip_extras': [],
                #try no regularization
                'no_reg': False,
                #search learning rates (False automatically use default)
                'search_lr': False
>>>>>>> 33b04cce36d2e37d6dd952fc8199a2021ff30530
        }
    else: #search space is big
        return {}

def lr_feature_selector():
    return {'model': 'LR',
    'regularizer': ['l1', 'l2'],
    #reg str
    'reg_strength': [-5, 5],
    'reg_strength_list': range(-5,6),
    #converges to
    'converge': [-10, -1],
    'converge_as_list': range(-10,0),
    'nmin_to_max_': [(1,1),(1,2),(1,3),(2,2),(2,3)],
    'binary_': [True, False],
    'use_idf_': [True, False],
    'st_wrd_': [None, 'english']}

def cnn_feats(model_num):
    return {'nmin_to_max_' + model_num: (1,1),
            'binary_' + model_num: False,
            'use_idf_' + model_num: False,
            'st_wrd_' + model_num: None}
