import random
from space import cnn_space

#random searches for a CNN
def get_cnn_model(model_num, search_space):
    space = cnn_space(search_space)

    param_dist = {'model_' + model_num: 'CNN',
            'word_vectors_' + model_num: ('word2vec', True),
            'delta_' + model_num: random.choice(space['delta_']),
            'flex_' + model_num: (True, random.random() * (space['flex_amt_'][1] -
                                space['flex_amt_'][1]) + space['flex_amt_'][0]),
            'dropout_' + model_num: random.random() * (space['dropout_'][1] -
                                space['dropout_'][0]) + space['dropout_'][0],
            'activation_fn_' + model_num: random.choice(space['activation_fn_']),
            'learning_rate_' + model_num: .0003,
            'filters_' + model_num: random.randint(*space['filters_']),
            'kernel_size_' + model_num: random.randint(*space['kernel_size_']),
            'kernel_increment_' + model_num: random.randint(*space['kernel_increment_']),
            'kernel_num_' + model_num: random.randint(*space['kernel_num_']),
            'batch_size_' + model_num: random.randint(*space['batch_size_'])
    }

    #regularize with option of no rig if no reg
    if space['no_reg'] == True:
        param_dist['regularizer_cnn_' + model_num] = random.choice([
            (None, 0.0),
            ('l2', (random.random() * (space['l2_'][1] - space['l2_'][0]) + space['l2_'][0])),
            ('l2_clip', (random.random() * (space['l2_clip_'][1] - space['l2_clip_'][0]) + space['l2_clip_'][0]))
        ])

    else:
        param_dist['regularizer_cnn_' + model_num] = random.choice([
            ('l2', (random.random() * (space['l2_'][1] - space['l2_'][0]) + space['l2_'][0])),
            ('l2_clip', (random.random() * (space['l2_clip_'][1] - space['l2_clip_'][0]) + space['l2_clip_'][0]))
        ])

    #overwrite previously set values to search over some parameters depending on options
    if space['search_lr'] == True:
        param_dist['learning_rate_' + model_num] = (random.lognormvariate(0, 1)) / 3000

    return param_dist

def get_feats(model_num):
    return {'nmin_to_max_' + model_num: random.choice([(1,1),(1,2),(1,3),(2,2),(2,3)]),
            'binary_' + model_num: random.choice([True, False]),
            'use_idf_' + model_num: random.choice([True, False]),
            'st_wrd_' + model_num: random.choice([None, 'english'])}
