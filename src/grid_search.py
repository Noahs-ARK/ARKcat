from feature_selector import *
import itertools, math
import random

#breaks when num_models > 1
#returns all combinations from a grid of hyperparameters
#searching lr not implemented yet
#return 25% and 75% percentile

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
            return grid_search('linear')
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
        feature_selector = lr_feature_selector()
        linear_list = [feature_selector['regularizer'],
            feature_selector['reg_strength_list'],
            feature_selector['converge_as_list'],
            feature_selector['nmin_to_max_'],
            feature_selector['binary_'],
            feature_selector['use_idf_'],
            feature_selector['st_wrd_']]
        self.enumerate_models_list = list(itertools.product(*linear_list))
        self.convert_to_dict()

    def get_cnn_model(self):
        feature_selector = cnn_feature_selector(self.search_space)
        general = [feature_selector['delta_'],
                    #flex
                    # [.15, .3],
                    self.to_list(feature_selector['flex_amt_'], feature_selector['grid'][2]),
                    self.to_list(feature_selector['filters_'], feature_selector['grid'][3]),
                    self.to_list(feature_selector['kernel_size_'], feature_selector['grid'][4]),
                    self.to_list(feature_selector['kernel_increment_'], feature_selector['grid'][5]),
                    self.to_list(feature_selector['kernel_num_'], feature_selector['grid'][6]),
                    self.to_list(feature_selector['dropout_'], feature_selector['grid'][7]),
                    self.to_list(feature_selector['batch_size_'], feature_selector['grid'][8]),
                    feature_selector['activation_fn_']]
                    # [.00017, .0015]
                    # [.0003]]
        if feature_selector['search_lr']:
            general.append([.00017, .000653])
        else:
            general.append([.0003])
        l2 = [['l2'], list(feature_selector['l2_'])] + general
        l2_clip = [['l2_clip'], list(feature_selector['l2_clip_'])] + general
        if feature_selector['no_reg']:
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
        if index == 1:
            try:
                return [float(sum(space))/len(space)]
            except TypeError:
                return [space]
        elif index == 2:
            quartile = float(sum(space))/len(space) - space[0] / 2
            return [space[0] + quartile, space[0] + 3 * quartile]
        else:
            space_list = [space[0], space[1]]
            space_range = space[1] - space[0]
            for option in range(index - 2):
                space_list.append(space[0] + option * space_range / (index - 1))
            return space_list

    def pop_model(self, num_models):
        models = self.grid.pop(0)
        for i in range(num_models-1):
            models.update(self.grid.pop())
        return models
