from hyperopt import fmin, tpe, hp, Trials, space_eval

def get_space(num_models, model_types):
    space = {}


    for i in range(num_models):
        add_model(str(i), space, model_types)
    return space

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

def get_cnn_model_big(model_num):
    hyperparams = {
            # choose btwn rand, word2vec--implement glove
            # 'word_vectors_' + model_num: hp.choice('word_vectors_' + model_num,[
            #     ('word2vec', hp.choice('word2vec_update_' + model_num, [True, False])),
            #     ('rand', hp.choice('rand_update_' + model_num, [True, False]))
            # ]),
            'model_' + model_num: 'CNN',
            'word_vector_update_' + model_num: hp.choice('word_vector_update_' + model_num, [True, False]),
            'delta_' + model_num: hp.choice('delta_' + model_num, [True, False]),
            'flex_' + model_num: hp.quniform('flex_' + model_num, 0, 15, 1),
            'filters_' + model_num: hp.quniform('filters_' + model_num, 10, 1000,1),
            # 'num_kernels_' + model_num: hp.quniform('num_kernels_' + model_num, 1, 5, 1),
            'kernel_size_1_' + model_num: hp.quniform('kernel_size_1_' + model_num, 1, 30, 1),
            'kernel_size_2_' + model_num: hp.quniform('kernel_size_2_' + model_num, 1, 30, 1),
            'kernel_size_3_' + model_num: hp.quniform('kernel_size_3_' + model_num, 1, 30, 1),
            'dropout_' + model_num: hp.uniform('dropout_' + model_num, 0, 1),
            'batch_size_' + model_num: hp.quniform('batch_size_' + model_num, 10, 200, 1),
            # iden, relu, and tanh
            'activation_fn_' + model_num: hp.choice('activation_fn_' + model_num, ['iden', 'relu', 'elu', 'tanh']),
            #none, clipped, or penalized
            'regularizer_cnn_' + model_num: hp.choice('regularizer_cnn_' + model_num, [
                (None, 0.0),
                ('l1', hp.loguniform('l1_strength_cnn_' + model_num, 0,10)),
                ('l2', hp.loguniform('l2_strength_cnn_' + model_num, 0,10)),
                ('l2_clip', hp.uniform('l2_clip_norm_' + model_num, 1,10))
            ]),
            'learning_rate_' + model_num: .0025 + (hp.lognormal('learning_rate_' + model_num, 0, 1) / 1000)
        }
    # doesn't work yet :(
    # print hyperparams['num_kernels_' + model_num]
    # for i in xrange(int(hyperparams['num_kernels_' + model_num])):
    #     hyperparams['kernel_' + i + '_' + model_num] = hp.quniform('kernel_size_' + i + '_'
    #                                                    + model_num, 1, 10, 1)

#can we make flex dependent on kernel size somehow?? or related to max len/len diff betwn batches
def get_cnn_model(model_num):
    hyperparams = {
            # choose btwn rand, word2vec--implement glove
            # 'word_vectors_' + model_num: hp.choice('word_vectors_' + model_num,[
            #     ('word2vec', hp.choice('word2vec_update_' + model_num, [True, False])),
            #     ('rand', hp.choice('rand_update_' + model_num, [True, False]))
            # ]),
            'model_' + model_num: 'CNN',
            'word_vector_update_' + model_num: hp.choice('word_vector_update_' + model_num, [True, False]),
            'delta_' + model_num: hp.choice('delta_' + model_num, [True, False]),
            'flex_' + model_num: hp.quniform('flex_' + model_num, 0, 15, 1),
            'filters_' + model_num: hp.quniform('filters_' + model_num, 100, 600,1),
            # 'num_kernels_' + model_num: hp.quniform('num_kernels_' + model_num, 1, 5, 1),
            'kernel_size_1_' + model_num: hp.quniform('kernel_size_1_' + model_num, 1, 30, 1),
            'kernel_size_2_' + model_num: hp.quniform('kernel_size_2_' + model_num, 1, 30, 1),
            'kernel_size_3_' + model_num: hp.quniform('kernel_size_3_' + model_num, 1, 30, 1),
            'dropout_' + model_num: hp.uniform('dropout_' + model_num, 0, 1),
            'batch_size_' + model_num: hp.quniform('batch_size_' + model_num, 10, 200, 1),
            # iden, relu, and tanh
            'activation_fn_' + model_num: hp.choice('activation_fn_' + model_num, ['iden', 'relu', 'elu', 'tanh']),
            #none, clipped, or penalized
            'regularizer_cnn_' + model_num: hp.choice('regularizer_cnn_' + model_num, [
                (None, 0.0),
                ('l2', hp.loguniform('l2_strength_cnn_' + model_num, 0,2)),
                ('l2_clip', hp.uniform('l2_clip_norm_' + model_num, 2,6))
            ]),
            'learning_rate_' + model_num: .0025 + (hp.lognormal('learning_rate_' + model_num, 0, 1) / 1000)
        }
    # doesn't work yet :(
    # print hyperparams['num_kernels_' + model_num]
    # for i in xrange(int(hyperparams['num_kernels_' + model_num])):
    #     hyperparams['kernel_' + i + '_' + model_num] = hp.quniform('kernel_size_' + i + '_'
    #                                                    + model_num, 1, 10, 1)
    return hyperparams

def add_model(model_num, space, model_types):
    set_of_models = []
    for m in model_types:
        if m == 'linear':
            set_of_models.append(get_linear_model(model_num))
        elif m == 'xgboost':
            set_of_models.append(get_xgboost_model(model_num))
        elif m == 'cnn':
            set_of_models.append(get_cnn_model(model_num))
        else:
            raise NameError('the model ' + m + ' is not implemented.')
    #DEBUGGING
    #if cnn then don't add binary option
    space['model_' + model_num] = hp.choice('model_' + model_num, set_of_models)
    space['features_' + model_num] = {
        'nmin_to_max_' + model_num: hp.choice('nmin_to_max_' + model_num,
                                              [(1,1),(1,2),(1,3),(2,2),(2,3)]),
        'binary_' + model_num: hp.choice('binary_' + model_num, [True, False]),
        'use_idf_' + model_num: hp.choice('transform_' + model_num, [True, False]),
        'st_wrd_' + model_num: hp.choice('st_word_' + model_num, [None, 'english'])
    }
