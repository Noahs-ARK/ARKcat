
from sklearn.linear_model import LogisticRegression as lr




class Model_LR:
    def __init__(self, params, n_labels):
        self.hp = params
        self.num_labels = n_labels

    def train(self, train_X, train_Y):
        self.model = lr(penalty=self.hp['regularizer'], C=self.hp['alpha'], tol=self.hp['converg_tol'])
        self.model.fit(train_X, train_Y)

    def predict(self, test_X):
        return self.model.predict(test_X)

    def predict_prob(self, test_X):
        return self.model.predict_proba(test_X)
