import os
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from model_xgb import Model_XGB
from model_lr import Model_LR
from model_cnn import Model_CNN
import re
#DEBUGGING
#import xgboost


class Data_and_Model_Manager:
    def __init__(self, f_and_p):
        self.feats_and_params = f_and_p
        self.trained_models = {}
        self.vectorizers = {}
        #DEBUGGING need to remove this train feature_dir thing, or use it
        self.train_feat_dirs = {}
        self.label_dict = {}
        self.num_labels = 0


    def init_model(self, params, n_labels, index_to_word = None):
        if params['model_type'] == 'LR':
            return Model_LR(params, n_labels)
        elif params['model_type'] == 'XGBoost':
            return Model_XGB(params, n_labels)
        elif params['model_type'] == 'CNN':
            return Model_CNN(params, n_labels, index_to_word)
        else:
            raise TypeError("you're trying to train this kind of model (which isn't implemented):" +
                            self.hp['model_type'])

    def read_data_and_labels(self, data_filename, label_filename):
        if not os.path.isfile(data_filename):
            return [], []
        data = []
        with open(data_filename, 'r') as input_file:
            for line in input_file:
                data.append(line.strip())

        labels = []
        with open(label_filename, 'r') as input_file:
            for line in input_file:
                labels.append(line.strip())
        return data,labels

    def load_train_and_dev_data(self, train_data_filename, train_label_filename,
                                    train_feature_dir, dev_data_filename, dev_label_filename,
                                    dev_feature_dir, verbose):
        train_x, train_y = self.read_data_and_labels(train_data_filename, train_label_filename)
        dev_x, dev_y = self.read_data_and_labels(dev_data_filename, dev_label_filename)
        self.train = [train_x, train_y]
        self.dev = [dev_x, dev_y]

    def k_fold_cv(self, num_folds):
        if num_folds == 1 and len(self.dev[0]) > 0:
            train_acc = self.train_models(self.train[0], self.train[1])
            dev_acc = self.predict_acc(self.dev[0], self.dev[1])
            return {'train_acc':train_acc, 'dev_acc':dev_acc}
        else:
            self.train[0].extend(self.dev[0])
            self.train[1].extend(self.dev[1])

            #PROBLEM: SELF.TRAIN NOT INITIIAL
            if num_folds < 5:
                folds = StratifiedKFold(self.train[1], 5, shuffle=True)
            else:
                folds = StratifiedKFold(self.train[1], num_folds, shuffle=True)
            avg_dev_acc = 0
            for train_indxs, dev_indxs in folds:

                cur_train_X = [self.train[0][i] for i in train_indxs]
                cur_train_Y = [self.train[1][i] for i in train_indxs]
                cur_dev_X = [self.train[0][i] for i in dev_indxs]
                cur_dev_Y = [self.train[1][i] for i in dev_indxs]
                self.train_models(cur_train_X, cur_train_Y)

                avg_dev_acc = avg_dev_acc + self.predict_acc(cur_dev_X, cur_dev_Y)/num_folds
            return {'train_acc':self.train_models(self.train[0], self.train[1]), 'dev_acc':avg_dev_acc}
    #eval wont work until I get test_X in correct format
    def train_models(self, train_X_raw, train_Y_raw):
        if len(train_X_raw) == 0:
            raise IOError("problem! the training set is empty.")

        probs = {}
        train_Y = self.convert_labels(train_Y_raw)
        for i, feat_and_param in self.feats_and_params.items():



            print("length of training data: ", len(train_X_raw))
            #DEBUGGING
            #Shouldn't have vectorizer initialized in two braches of this if statement
            #should have the bayes opt know not to use n-grams for cnn
            if feat_and_param['params']['model_type'] == 'CNN':

                feat_and_param['feats']['ngram_range'] = (1,1)
                feat_and_param['feats']['binary'] = False
                vectorizer = TfidfVectorizer(**feat_and_param['feats'])
                vectorizer.fit(train_X_raw)
                tokenizer = TfidfVectorizer.build_tokenizer(vectorizer)
                train_X_raw_tokenized = [tokenizer(ex) for ex in train_X_raw]
                print("size of tokenized: ", len(train_X_raw_tokenized))
                train_X = []
                for example in train_X_raw_tokenized:
                    for i in range(len(example)):
                        example[i] = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", example[i])
                    train_X.append([vectorizer.transform(example)])

                index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
                cur_model = self.init_model(feat_and_param['params'], self.num_labels, index_to_word)
            else:
                vectorizer = TfidfVectorizer(**feat_and_param['feats'])
                vectorizer.fit(train_X_raw)
                train_X = vectorizer.transform(train_X_raw)
                cur_model = self.init_model(feat_and_param['params'], self.num_labels)


            cur_model.train(train_X, train_Y)
            self.trained_models[i] = cur_model
            self.vectorizers[i] = vectorizer
            probs[i] = cur_model.predict_prob(train_X)
        preds = self.convert_probs_to_preds(probs)
        return metrics.accuracy_score(train_Y, preds)


    def convert_labels(self, train_Y_old):
        new_Y = []
        for y in train_Y_old:
            if y not in self.label_dict:
                self.label_dict[y] = self.num_labels
                self.label_dict[self.num_labels] = y
                self.num_labels = self.num_labels + 1
            new_Y.append(self.label_dict[y])
        return new_Y

    def predict_acc(self, test_X_raw, test_Y):
        pred_probs = {}
        for i, feat_and_param in self.feats_and_params.items():
            test_X = self.vectorizers[i].transform(test_X_raw)
            print("SIZE OF TEST_X:")
            print(test_X.shape)
            pred_probs[i] = self.trained_models[i].predict_prob(test_X)

        preds_as_nums = self.convert_probs_to_preds(pred_probs)
        preds = self.convert_labels(preds_as_nums)
        return metrics.accuracy_score(test_Y, preds)

    def predict_acc_from_file(self, test_X_filename, test_Y_filename):
        test_X, test_Y = self.read_data_and_labels(test_X_filename, test_Y_filename)
        return self.predict_acc(test_X, test_Y)

    def convert_probs_to_preds(self, probs):
        preds = []
        for i in range(len(probs[0])):
            cur_probs = []
            for k in range(self.num_labels):
                cur_probs.append(0)
            for j in range(len(probs)):
                for k in range(self.num_labels):
                    cur_probs[k] = probs[j][i][k]/len(probs) + cur_probs[k]
            preds.append(cur_probs.index(max(cur_probs)))
        return preds
