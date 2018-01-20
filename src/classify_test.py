from hyperopt import STATUS_OK
from models_and_data import Data_and_Model_Manager



def classify(train_data_filename, train_label_filename, dev_data_filename, dev_label_filename,
             train_feature_dir, dev_feature_dir, model_dir, word2vec_filename, feats_and_params,
             verbose=1, folds=-1):
    m_and_d = Data_and_Model_Manager(feats_and_params, model_dir, word2vec_filename)
    m_and_d.load_train_and_dev_data(train_data_filename, train_label_filename,
                                    train_feature_dir, dev_data_filename, dev_label_filename,
                                    dev_feature_dir, verbose)
    acc = m_and_d.k_fold_cv(folds)

    print('train acc: ' + str(acc['train_acc']))
    print('dev acc: ' + str(acc['dev_acc']))
    return {'loss': -acc['dev_acc'], 'status': STATUS_OK, 'model': m_and_d}

