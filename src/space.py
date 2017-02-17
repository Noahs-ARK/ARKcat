'''
Hyperparameter explanation:

NOTE: features that are ranges must be specified as such even if they are intended to be specified to one value

delta: trainable weight matrix that is multiplied by the word vectors (options: True, False)
flex: parameter to randomly pad examples in order to increase variability in batching
    between training epochs. (False, 0.0) for no padding; (True, float in range (0, 1))
    for amount of additional padding to add as a multiple of maximum example length in the training data.
    There is a 15% chance of adding each of flex_amt * max_length / max_length or half that amount to the
    beginning of the example and an equivalent, independent probability of padding the end.
filters: number of convolutional filters per kernel; second dimension of the output of the convolutional
    layer. (Higher numbers of filters preserve more information from the word vectors)
kernel size: number of tokens "seen" by smallest kernel
kernel increment: how much larger each kernel is, i.e. if kernel size is 3 and increment is 1
    than the smallest 2 kernels would be 3 and 4
kernel number: number of kernels
dropout: chance of dropping out any neuron in the output of max pooling (note: no dropout in the convolutional layer!)
batch size: number of examples processed concurrently as one tensor during training. (During eval, examples are evaluated singly.)
activation fn: type of activation fn. Currently implemented: 'iden', identity; 'relu', rectified linear unit;
    'elu', exponential linear unit; 'tanh', hyperbolic tangent; 'sigmoid'.
l2: 10^x is the l2 regularization if l2 is chosen during hyperparameter selection
l2_clip: norm of l2 clipping if l2 clipping is chosen during hyperparameter selection
no_reg: include no regularization whatsoever in search (True or False)
search_lr: search learning rate (True or False). If true, uses a logarithmic normal distribution with mu = 0, stdev = 1 divided by 3000
grid: size of grid for each hyperparameter. For example, 4 at the index of dropout would cause the grid
    to choose four options within the range specified for dropout.
'''

def cnn_space(search_space):
    if search_space == 'reg':
        return {'model_type_': 'CNN',
                'delta_': [False],
                'flex_amt_': (0.15, 0.15),
                'filters_': (100, 100),
                'kernel_size_': (3, 3),
                'kernel_increment_': (1, 1),
                'kernel_num_': (3, 3),
                'dropout_': (0, 0.75),
                'batch_size_': (50, 50),
                'activation_fn_': ['relu'],
                'l2_': (-5.0, -1.0),
                'l2_clip_': (1.0, 5.0),
                'no_reg': True,
                'search_lr': True,
                'grid': [1,1,1,1,1,1,1,4,1,1,4,3]
        }
    elif search_space == 'arch':
        return {'model_type_': 'CNN',
                'delta_': [False],#not implemented
                'flex_amt_': (0.0,0.3),
                'filters_': (100,600),
                'kernel_size_': (2,10),
                'kernel_increment_': (0,3),
                'kernel_num_': (3,5),
                'dropout_': (0.25,0.75),
                'batch_size_': (50,50),
                'activation_fn_': ['iden', 'relu', 'elu'],
                #not sure
                'l2_': (-8.0,-2.0),
                'l2_clip_': (2.0,10.0),
                'no_reg': False,
                'search_lr': False,
                'grid': [1,2,2,2,2,2,2,2,2,3,2,2]
        }
    elif search_space == 'big':
        return {'model_type_': 'CNN',
                'delta_': [False],#not implemented
                'flex_amt_': (0.0,0.3),
                'filters_': (100,600),
                'kernel_size_': (2,30),
                'kernel_increment_': (0,5),
                'kernel_num_': (1,8),
                'dropout_': (0,1),
                'batch_size_': (10,200),
                'activation_fn_': ['iden', 'relu', 'elu', 'tanh', 'sigmoid'],
                'l2_': (-8.0,-2.0),
                'l2_clip_': (2.0,10.0),
                'no_reg': True,
                'search_lr': True,
                'grid': [1,2,2,2,2,2,2,2,2,5,2,2]}
    else:
        print 'search space not implemented for CNN'
        raise NotImplementedError

def lr_space():
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
