import os,sys
import xgboost
import classify_test
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression as lr


class Model:
    def __init__(self, params):
        self.hp = params
        
        
    def train(self, train_X, train_Y):
        
        if self.hp['model_type'] == 'LR' or self.hp['model_type'] == 'SVM':
            self.train_linear(train_X, train_Y)
        elif self.hp['model_type'] == 'XGBoost':
            self.train_xgboost(train_X, train_Y)
        else:
            #should move this error up to the init
            sys.exit('Model type ' + self.hp['model_type'] + ' not supported')

    def train_xgboost(self, train_X, train_Y):
        
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
                 'lambda':self.hp['lambda']}
        
        self.model = xgboost.train(param, dtrain, self.hp['num_round'])

            

    def train_linear(self, train_X, train_Y):
        if self.hp['model_type'] == 'LR':
            self.model = lr(penalty=self.hp['regularizer'], C=self.hp['alpha'], tol=self.hp['converg_tol'])
        else:
            self.model = svm.LinearSVC(
                penalty=self.hp['regularizer'], C=self.hp['alpha'], tol=self.hp['converg_tol'])
        self.model.fit(train_X, train_Y)



    def predict(self, test_X):
        if self.hp['model_type'] == 'SVM' or self.hp['model_type'] == 'LR':
            return self.predict_linear(test_X)
        if self.hp['model_type'] == 'XGBoost':
            return self.predict_xgboost(test_X)


    def predict_linear(self, test_X):
        return self.model.predict(test_X)
    

    def predict_xgboost(self, test_X):
        dtest = xgboost.DMatrix(test_X)
        test_pred = self.model.predict(dtest)
        test_pred_round = []
        [test_pred_round.append(int(round(pred))) for pred in test_pred]
        return test_pred_round
