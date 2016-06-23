import xgboost
import os,sys
import scipy


class Model_XGB:
    def __init__(self, params, n_labels):
        self.hp = params
        self.num_labels = n_labels

    def train(self, train_X, train_Y):        
        print('about to try to make DMatrix')
        dtrain = xgboost.DMatrix(train_X, label=train_Y)
        print('now printing the hyperparams:')
        print(self.hp)
        
        if self.hp['regularizer'] == 'l1':
            self.hp['lambda'] = 0
            self.hp['alpha'] = self.hp['reg_strength']
        elif self.hp['regularizer'] == 'l2':
            self.hp['alpha'] = 0
            self.hp['lambda'] = self.hp['reg_strength']
        param = {'eta':self.hp['eta'],
                 'gamma':self.hp['gamma'],
                 'max_depth':self.hp['max_depth'],
                 'min_child_weight':self.hp['min_child_weight'],
                 'max_delta_step':self.hp['max_delta_step'],
                 'subsample':self.hp['subsample'],
                 'alpha':self.hp['alpha'],
                 'lambda':self.hp['lambda'],
                 'objective':'multi:softprob',
                 'num_class':self.num_labels
        }
        
        self.model = xgboost.train(param, dtrain, self.hp['num_round'])

    def predict(self, test_X):
        test_pred = self.predict_prob(test_X)
        test_pred_round = []
        [test_pred_round.append(int(round(pred))) for pred in test_pred]
        return test_pred_round

    def predict_prob(self, test_X):
        test_X = scipy.sparse.csc_matrix(test_X)
        dtest = xgboost.DMatrix(test_X)
        return  self.model.predict(dtest)

