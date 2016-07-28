def cnn_feature_selector():
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
            'l2_': (-8,-2),
            'l2_clip_': (2,10),
            'l2_extras': [],
            'clip_extras': []
    }

def yoon_kim_cnn():
    return {'model_': 'CNN',
            'delta_': [False],
            'flex_amt_': (0.15),
            'filters_': (100),
            'kernel_size_': (3),
            'kernel_increment_': (1),
            'kernel_num_': (3),
            'dropout_': (0, 0.25, 0.5, 0.75),
            'batch_size_': (50),
            'activation_fn_': ['relu'],
            'l2_': (-4, -1),
            'l2_clip_': (1,5),
            'l2_extras': [-3, -2],
            'clip_extras': [3]
    }


def lr_feature_selector():
    return {'model': 'LR',
    'regularizer': ['l1', 'l2'],
    #reg str
    'reg_strength': [-5, 5],
    'reg_strength_list': range(-5,6),
    #converges to
    'converge': [-10, -1],
    'converge_as_list': range(-10,0)}
