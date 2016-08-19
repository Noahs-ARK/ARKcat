from space import *
import itertools, math
import random

#returns all combinations from a grid of hyperparameters

def get_grid(model_types, search_space):
        if model_types[0] == 'cnn':
            for model in model_types:
                if model != 'cnn':
                    print 'model types don\'t match'
                    raise TypeError
            return grid_search('cnn', search_space)
        elif model_types[0] == 'linear':
            for model in model_types:
                if model != 'linear':
                    print 'model types don\'t match'
                    raise TypeError
            print grid_search('linear', search_space)
            return grid_search('linear', search_space)
        else:
            print 'Grid search only implemented for CNN and LR'
            raise NotImplementedError

class grid_search():

    #if all models are CNN...
    def __init__(self, model_type, search_space=None):
        self.model_type = model_type
        self.search_space = search_space
        if self.model_type == 'cnn':
            self.get_cnn_model()
        elif self.model_type == 'linear':
            self.get_linear_model()
        else:
            raise NotImplementedError

    def get_linear_model(self):
        space = lr_space(self.search_space)
        linear_list = [space['regularizer'],
            space['reg_strength_list'],
            space['converge_as_list'],
            space['nmin_to_max_'],
            space['binary_'],
            space['use_idf_'],
            space['st_wrd_']]
        self.enumerate_models_list = list(itertools.product(*linear_list))
        self.convert_to_dict()

    def get_cnn_model(self):
        space = cnn_space(self.search_space)
        general = [space['delta_'],
                    self.to_list(space['flex_amt_'], space['grid'][2]),
                    self.to_list(space['filters_'], space['grid'][3]),
                    self.to_list(space['kernel_size_'], space['grid'][4]),
                    self.to_list(space['kernel_increment_'], space['grid'][5]),
                    self.to_list(space['kernel_num_'], space['grid'][6]),
                    self.to_list(space['dropout_'], space['grid'][7]),
                    self.to_list(space['batch_size_'], space['grid'][8]),
                    space['activation_fn_']]
        if space['search_lr']:
            general.append([.00017, .000653])
        else:
            general.append([.0003])
        l2 = [['l2'], list(space['l2_'])] + general
        l2_clip = [['l2_clip'], list(space['l2_clip_'])] + general
        print "l2", l2
        print "l3_cli", l2_clip
        if space['no_reg']:
            no_reg = [[None], [0.0]] + general
            self.enumerate_models_list = list(itertools.product(*l2)) + list(itertools.product(*l2_clip)) + list(itertools.product(*no_reg))
        else:
            self.enumerate_models_list = list(itertools.product(*l2)) + list(itertools.product(*l2_clip))
        self.convert_to_dict()

    def convert_to_dict(self):
        model_counter = 0
        list_of_models = []
        for model in self.enumerate_models_list:
            model_num = str(0)
            if self.model_type == 'cnn':
                hyperparam_grid = self.cnn_get_grid(model_num, model)
            else:
                hyperparam_grid = self.linear_get_grid(model_num, model)
            model_counter += 1
            list_of_models.append(hyperparam_grid)
        self.grid = list_of_models

    def cnn_get_grid(self, model_num, model):
        return {'model_' + model_num:
                    {'model_' + model_num: 'CNN',
                    'word_vectors_' + model_num: ('word2vec', True),
                    'regularizer_cnn_' + model_num: (model[0], model[1]),
                    'delta_' + model_num: model[2],
                    'flex_' + model_num: (True, model[3]),
                    'filters_' + model_num: model[4],
                    'kernel_size_' + model_num: model[5],
                    'kernel_increment_' + model_num: model[6],
                    'kernel_num_' + model_num: model[7],
                    'dropout_' + model_num: model[8],
                    'batch_size_' + model_num: model[9],
                    # iden, relu, and tanh
                    'activation_fn_' + model_num: model[10],
                    'learning_rate_' + model_num: model[11]},
            'features_' + model_num: cnn_feats(model_num)}

    def linear_get_grid(self, model_num, model):
        return {'model_' + model_num:
                {'model_' + model_num: 'LR',
                'regularizer_lr_' + model_num: (model[0], math.exp(model[1])),
                'converg_tol_' + model_num: math.exp(model[2])},
            'features_' + model_num: {'nmin_to_max_' + model_num: model[3],
                    'binary_' + model_num: model[4],
                    'use_idf_' + model_num: model[5],
                    'st_wrd_' + model_num: model[6]}}

    def to_list(self, space, index):
        if type(space) != tuple:
            return [space]
        elif space[0] == space[1]:
            return [space[0]]

        if index == 1:
            return [float(sum(space))/len(space)]
        elif index == 2:
            quartile = float(sum(space))/len(space) - space[0] / 2
            return [space[0] + quartile, space[0] + 3 * quartile]
        else:
            space_list = [space[0], space[1]]
            space_range = space[1] - space[0]
            print space_range
            for option in range(index - 2):
                print space[0] + option * space_range / (index - 1)
                space_list.append(space[0] + option * space_range / (index - 1))
            return space_list

    def pop_model(self, num_models):
        models = self.grid.pop(0)
        for i in range(num_models-1):
            models.update(self.grid.pop())
        return models
