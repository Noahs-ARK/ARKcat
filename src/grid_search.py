from feature_selector import cnn_feature_selector, lr_feature_selector
import itertools, math
#returns all combinations from a grid of hyperparameters

def get_grid(model_types):
        if model_types[0] == 'cnn':
            for model in model_types:
                if model != 'cnn':
                    print 'model types don\'t match'
                    raise TypeError
            return grid_search('cnn')
        elif model_types[0] == 'linear':
            for model in model_types:
                if model != 'linear':
                    print 'model types don\'t match'
                    raise TypeError
            return grid_search('linear')
        else:
            print 'Grid search only implemented for CNN and LogReg'
            raise NotImplementedError

class grid_search():

    #if all models are CNN...
    def __init__(self, model_type):
        self.model_type = model_type
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
        feature_selector = cnn_feature_selector()
        general = [feature_selector['delta_'],
                    #flex
                    # [.15, .3],
                    list(feature_selector['flex_amt_']),
                    list(feature_selector['filters_']),
                    list(feature_selector['kernel_size_']),
                    list(feature_selector['kernel_increment_']),
                    list(feature_selector['kernel_num_']),
                    list(feature_selector['dropout_']),
                    list(feature_selector['batch_size_']),
                    feature_selector['activation_fn_'],
                    #learning_rate
                    # [.00075, .0015]
                    [.001]]
        l2 = [['l2'], list(feature_selector['l2_']) + feature_selector['l2_extras']] + general
        l2_clip = [['l2_clip'], list(feature_selector['l2_clip_']) + feature_selector['clip_extras']] + general
        if feature_selector['no_reg']:
            no_reg = [[None] + [0.0] + general]
            self.enumerate_models_list = list(itertools.product(*l2)) + list(itertools.product(*l2_clip)) + list(itertools.product(*no_reg))
        else:
            self.enumerate_models_list = list(itertools.product(*l2)) + list(itertools.product(*l2_clip))
        self.convert_to_dict()

    def convert_to_dict(self):
        model_counter = 0
        list_of_models = []
        for model in self.enumerate_models_list:
            model_num = str(model_counter)
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
            'features_' + model_num: self.get_feats(model_num)}

    def linear_get_grid(self, model_num, model):
        return {'model_' + model_num:
                {'model_' + model_num: 'LR',
                'regularizer_lr_' + model_num: (model[0], math.exp(model[1])),
                'converg_tol_' + model_num: math.exp(model[2])},
            'features_' + model_num: {'nmin_to_max_' + model_num: model[3],
                    'binary_' + model_num: model[4],
                    'use_idf_' + model_num: model[5],
                    'st_wrd_' + model_num: model[6]}}

    def get_feats(self, model_num):
        if self.model_type == 'cnn':
            return {'nmin_to_max_' + model_num: (1,1),
                    'binary_' + model_num: False,
                    'use_idf_' + model_num: False,
                    'st_wrd_' + model_num: None}

    def pop_model(self, num_models):
        models = self.grid.pop(0)
        for i in range(num_models-1):
            models.update(self.grid.pop())
        return models
