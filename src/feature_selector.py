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
            'l2_clip_': (2,10)
}

def lr_feature_selector():
    return {}
