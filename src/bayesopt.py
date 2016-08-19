from hyperopt import fmin, tpe, hp, Trials, space_eval
from space import cnn_space

#uses Bayesian optimization to choose parameters of model

def get_linear_model(model_num):
    return {
        'model_' + model_num: 'LR',
        'regularizer_lr_' + model_num: hp.choice('regularizer_lr_' + model_num,[
            ('l1', hp.loguniform('l1_strength_lr_' + model_num, -5,5)),
            ('l2', hp.loguniform('l2_strength_lr_' + model_num, -5,5))
        ]),
        'converg_tol_' + model_num: hp.loguniform('converg_tol_' + model_num, -10, -1)
    }

def get_xgboost_model(model_num):
    return {
            'model_' + model_num: 'XGBoost',
            'eta_' + model_num: hp.loguniform('eta_' + model_num,-5,0),
            'gamma_' + model_num: hp.uniform('gamma_' + model_num,0,10),
            'max_depth_' + model_num: hp.quniform('max_depth_' + model_num, 1,30,1),
            'min_child_weight_' + model_num: hp.uniform('min_child_weight_' + model_num, 0, 10),
            'max_delta_step_' + model_num: hp.uniform('max_delta_step_' + model_num, 0, 10),
            'num_round_' + model_num: hp.quniform('num_round_' + model_num, 1, 10, 1),
            'subsample_' + model_num: 1,
            'regularizer_xgb_' + model_num: hp.choice('regularizer_xgb_' + model_num,[
                ('l1', hp.loguniform('l1_strength_xgb_' + model_num, -5,5)),
                ('l2', hp.loguniform('l2_strength_xgb_' + model_num, -5,5))
            ])
        }

def get_cnn_model(model_num, search_space):
    space = cnn_space(search_space)
    hparams = {'model_' + model_num: 'CNN',
            'word_vectors_' + model_num: ('word2vec', True),
            'delta_' + model_num: True,
            'flex_' + model_num: (True, .15),
            'filters_' + model_num: hp.quniform('filters_' + model_num, *space['filters_'], 1),
            'kernel_size_' + model_num: hp.quniform('kernel_size_' + model_num, *space['kernel_size_'], 1),
            'kernel_increment_' + model_num: hp.quniform('kernel_increment_' + model_num, *space['kernel_increment_'], 1),
            'kernel_num_' + model_num: hp.quniform('kernel_num_' + model_num, *space['kernel_num_'], 1),
            'dropout_' + model_num: hp.uniform('dropout_' + model_num, *space['dropout_']),
            'batch_size_' + model_num: hp.quniform('batch_size_' + model_num, *space['batch_size_'], 1),
            'activation_fn_' + model_num: hp.choice('activation_fn_' + model_num, space['activation_fn_'])}

    if space['no_reg']:
        hparams['regularizer_cnn_' + model_num] = hp.choice('regularizer_cnn_' + model_num, [
                (None, 0.0),
                ('l2', hp.uniform('l2_strength_cnn_' + model_num, *space['l2_'])),
                ('l2_clip', hp.uniform('l2_clip_norm_' + model_num, *space['l2_clip_']))
            ])

    else:
        hparams['regularizer_cnn_' + model_num] = hp.choice('regularizer_cnn_' + model_num, [
                ('l2', hp.uniform('l2_strength_cnn_' + model_num, *space['l2_'])),
                ('l2_clip', hp.uniform('l2_clip_norm_' + model_num, *space['l2_clip_']))
            ])

    if space['search_lr']:
        hparams['learning_rate_' + model_num] = hp.lognormal('learning_rate_' + model_num, 0, 1) / 3000
    else:
        hparams['learning_rate_' + model_num] = .0003

def get_feats(model_num):
    return {'nmin_to_max_' + model_num: hp.choice('nmin_to_max_' + model_num,
                                                  [(1,1),(1,2),(1,3),(2,2),(2,3)]),
            'binary_' + model_num: hp.choice('binary_' + model_num, [True, False]),
            'use_idf_' + model_num: hp.choice('transform_' + model_num, [True, False]),
            'st_wrd_' + model_num: hp.choice('st_word_' + model_num, [None, 'english'])}
