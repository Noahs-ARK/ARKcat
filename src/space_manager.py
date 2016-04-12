from hyperopt import fmin, tpe, hp, Trials, space_eval

def get_space(num_models):
    space = {}

    for i in range(num_models):
        add_model(str(i), space)
    return space

def get_linear_model(model_num):
    return {
        'model_' + model_num: 'LR',
        'regularizer_lr_' + model_num: hp.choice('regularizer_lr_' + model_num,[
        ('l1', hp.uniform('l1_strength_lr_' + model_num, 0,1)),
            ('l2', hp.uniform('l2_strength_lr_' + model_num, 0,1))
            
#                    ('l1', hp.loguniform('l1_strength', np.log(1e-7), np.log(10**2))),
#                    ('l2', hp.loguniform('l2_strength', np.log(1e-7), np.log(100)))
        ]),
        'converg_tol_' + model_num: hp.loguniform('converg_tol_' + model_num, -10, -1)
    }

def get_xgboost_model(model_num):
    return {
            'model_' + model_num: 'XGBoost',
            'eta_' + model_num: hp.uniform('eta_' + model_num,0,1),
            'gamma_' + model_num: hp.uniform('gamma_' + model_num,0,10),
            'max_depth_' + model_num: hp.quniform('max_depth_' + model_num, 1,50,1),
            'min_child_weight_' + model_num: hp.uniform('min_child_weight_' + model_num, 0, 10),
            'max_delta_step_' + model_num: hp.uniform('max_delta_step_' + model_num, 0, 10),
            'num_round_' + model_num: hp.quniform('num_round_' + model_num, 1, 10, 1),
            'subsample_' + model_num: hp.uniform('subsample_' + model_num, .001, 1),
            'regularizer_xgb_' + model_num: hp.choice('regularizer_xgb_' + model_num,[
                ('l1', hp.uniform('l1_strength_xgb_' + model_num, 0,1)),
                ('l2', hp.uniform('l2_strength_xgb_' + model_num, 0,1))
                
#                    ('l1_' + model_num, hp.loguniform('l1_strength_' + model_num, np.log(1e-7), np.log(10**2))),
#                    ('l2_' + model_num, hp.loguniform('l2_strength_' + model_num, np.log(1e-7), np.log(100)))
            ])
        }


def add_model(model_num, space):
    set_of_models = [get_xgboost_model(model_num)]
    space['model_' + model_num] = hp.choice('model_' + model_num, set_of_models)
    space['features_' + model_num] = {
        'unigrams_' + model_num:
        {
            'transform_' + model_num: hp.choice('u_transform_' + model_num, ['None', 'binarize', 'tfidf']),
            'min_df_' + model_num: hp.choice('u_min_df_' + model_num,[1,2,3,4,5])
        },
        'bigrams_' + model_num:
        hp.choice('bigrams_' + model_num, [
            {
                'use_' + model_num: False
            },
            {
                'use_' + model_num: True,
                'transform_' + model_num: hp.choice('b_transform_' + model_num, ['None', 'binarize', 'tfidf']),
                'min_df_' + model_num: hp.choice('b_min_df_' + model_num,[1,2,3,4,5])
            }
        ]),
    }
