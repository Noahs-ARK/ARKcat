import numpy as np
import tensorflow as tf
import cnn_train
import cnn_eval
from cnn_methods import *
import os,sys
import scipy

class Model_CNN:
    def __init__(self, params, n_labels, indices_to_words, model_dir):
        self.hp = params
        self.num_labels = n_labels
        self.indices_to_words = indices_to_words
        self.model_dir = model_dir

    #also need dev set--split off in here??
    def train(self, train_X, train_Y):
        print 'train_X type:', type(train_X[0][0])
        print 'still needs debug!!!'
        self.params = {
                'model_num' : self.hp['model_num'],
                'FLEX' : self.hp['flex'],
                'FILTERS' : self.hp['filters'],
                'FILTERS' : 1,
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
                #'UPDATE_WORD_VECS' : self.hp['word_vector_update'],
                'UPDATE_WORD_VECS' : False,
                'USE_DELTA' : False,
                'KERNEL_SIZES' : [self.hp['kernel_size_1'],
                                  self.hp['kernel_size_2'],
                                  self.hp['kernel_size_3']],
                # 'KERNEL_SIZES' : self.hp['kernel_sizes']
                #'USE_DELTA' : self.hp['delta'],

                'WORD_VECTOR_LENGTH' : 300,
                'CLASSES' : self.num_labels,
                'EPOCHS' : 15,
                #set by program-do not change!
                'epoch' : 1,
                'l2-loss' : tf.constant(0)
        }
        self.key_array, self.vocab = dict_to_array(self.indices_to_words, self.params)
        train_X, self.params['MAX_LENGTH'] = to_dense(train_X)
        train_Y = one_hot(train_Y, self.params['CLASSES'])
        self.model = cnn_train.main(self.params, train_X, train_Y, self.key_array, self.model_dir)

    #one-hot array
    def predict(self, test_X, measure = 'predict'):
        print 'test_X type:', type(test_X[0])
        test_X, self.params['MAX_LENGTH'] = to_dense(test_X)
        return cnn_eval.main(self.model, self.params, test_X, self.key_array,
                        measure)

    #softmax array
    def predict_prob(self, test_X):
        return self.predict(test_X, measure = 'scores')
