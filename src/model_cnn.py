import numpy as np
import tensorflow as tf
import cnn_train
import text_cnn_methods_temp
import os,sys
import scipy

class Model_CNN:
    def __init__(self, params, n_labels, indices_to_words):
        self.hp = params
        self.num_labels = n_labels
        self.indices_to_words = indices_to_words

    #also need dev set--split off in here??
    def train(self, train_X, train_Y):

        self.params = {
                'FLEX' : self.hp['flex'],
                'FILTERS' : self.hp['filters'],
                'ACTIVATION_FN' : self.hp['activation_fn'],
                'REGULARIZER' : self.hp['regularizer'],
                'REG_STRENGTH' : self.hp['reg_strength'],
                'TRAIN_DROPOUT' : self.hp['dropout'],
                'BATCH_SIZE' : self.hp['batch_size'],
                # 'optimizer_' + model_num: 1,# hp.choice('optimizer_' + model_num, 0, 1),
                'LEARNING_RATE' : self.hp['learning_rate'],
                #not implemented
                #'USE_WORD2VEC' : self.hp['use_word2vec'],
                'USE_WORD2VEC' : False,
                'UPDATE_WORD_VECS' : self.hp['word_vector_update'],
                'USE_DELTA' : self.hp['delta'],

                'WORD_VECTOR_LENGTH' : 300,
                'CLASSES' : self.num_labels,
                'EPOCHS' : 15,
                #set by program-do not change!
                'epoch' : 1,
                'l2-loss' : tf.constant(0),
                'KERNEL_SIZES' : [self.hp['kernel_size_1'],
                                  self.hp['kernel_size_2'],
                                  self.hp['kernel_size_3']]
        }
        if self.hp['word_vector_update'] == 0:
            self.params['UPDATE_WORD_VECS'] = False
        self.key_array = cnn_train.dict_to_array(self.indices_to_words, self.params)
        train_X, self.params['MAX_LENGTH'] = cnn_train.to_dense(train_X)
        train_Y = cnn_train.one_hot(train_Y, self.params['CLASSES'])
        val_split = len(train_X)/10
        val_X = train_X[val_split:]
        val_Y = train_Y[val_split:]
        train_X = train_X[:val_split]
        train_Y = train_Y[:val_split]
        print 'trainxcheck', type(train_X[0]), train_X[0]
        print 'valxcheck', type(val_X[0]), val_X[0]
        self.model = cnn_train.main(self.params, train_X, train_Y, val_X, val_Y, self.key_array)

    #one-hot array
    def predict(self, test_X, prob = False):
        test_bundle, self.vocab, self.embed_keys, new_key_array = cnn_eval.get_test_data(test_X, self.vocab, self.embed_keys, self.key_array, self.params)
        self.key_array = np.concat((key_array, new_key_array))
        return cnn_eval(self.model, self.params, test_bundle, new_key_array,
                        prob = prob)

    #softmax array
    def predict_prob(self, test_X):
        return predict(test_X, prob = True)
