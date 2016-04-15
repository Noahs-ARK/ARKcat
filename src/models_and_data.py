
import classify_test
from models import Model
from sklearn import metrics


class Data_and_Model_Manager:
    def __init__(self, f_and_p):
        self.feats_and_params = f_and_p
        self.trained_models = {}
        self.vectorizers = {}
        self.train_feat_dirs = {}
        

    def train_models(self, train_data_filename, train_label_filename, train_feature_dir, verbose):
        probs = {}
        for i, feat_and_param in self.feats_and_params.items():
            
            cur_model = Model(feat_and_param['params'])
            train_X, train_Y, vectorizer = classify_test.load_features(train_data_filename, 
                                                           train_label_filename, train_feature_dir,
                                                           feat_and_param['feats'], verbose)
            cur_model.train(train_X, train_Y)
            self.trained_models[i] = cur_model
            self.vectorizers[i] = vectorizer
            probs[i] = cur_model.predict_prob(train_X)
            #DEBUGGING need to remove this train feature_dir thing, or use it
            self.train_feat_dirs[i] = train_feature_dir
        preds = self.convert_probs_to_preds(probs)
        return metrics.accuracy_score(train_Y, preds)
            

    def predict_acc(self, data_filename, label_filename, feature_dir, verbose):
        pred_probs = {}
        for i, feat_and_param in self.feats_and_params.items():
            test_X, test_Y = classify_test.load_features(data_filename, label_filename, feature_dir, 
                                           feat_and_param['feats'], verbose, 
                                                         vectorizer=self.vectorizers[i])
            pred_probs[i] = self.trained_models[i].predict_prob(test_X)
        preds = self.convert_probs_to_preds(pred_probs)
        return metrics.accuracy_score(test_Y, preds)


    def convert_probs_to_preds(self, probs):
        preds = []
        for i in range(len(probs[0])):
            preds.append(0)
            for j in range(len(probs)):
                preds[i] = preds[i] + probs[j][i]/len(probs)
            preds[i] = str(int(round(preds[i])))
        
        return preds
