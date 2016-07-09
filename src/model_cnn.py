import numpy as np
import tensorflow as tf
import cnn_train
from cnn_methods import *
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
                'model_num' : self.hp['model_num'],
                'FLEX' : self.hp['flex'],
                'FILTERS' : self.hp['filters'],
                #'FILTERS' : 10,
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
                'UPDATE_WORD_VECS' : True,
                'USE_DELTA' : True,
                'KERNEL_SIZES' : [self.hp['kernel_size_1'],
                                  self.hp['kernel_size_2'],
                                  self.hp['kernel_size_3']],
                #'USE_DELTA' : self.hp['delta'],

                'WORD_VECTOR_LENGTH' : 300,
                'CLASSES' : self.num_labels,
                'EPOCHS' : 15,
                #set by program-do not change!
                'epoch' : 1,
                'l2-loss' : tf.constant(0)
        }
        if self.hp['word_vector_update'] == 0:
            self.params['UPDATE_WORD_VECS'] = False
        self.vocab = process(self.indices_to_words.itervalues())
        self.key_array = dict_to_array(self.vocab, self.params)
        train_X, self.params['MAX_LENGTH'] = to_dense(train_X)
        train_Y = one_hot(train_Y, self.params['CLASSES'])
        print 'debug 0'
        self.model, self.key_array, self.delta = cnn_train.main(self.params, train_X, train_Y, self.key_array)
        print 'debug key array', type(self.key_array)

    #one-hot array
    def predict(self, test_X, indices_to_words, measure = 'predict'):
        self.key_array, test_vocab_key, self.vocab = update_vocab(self.key_array,
                                                                  self.vocab,
                                                                  indices_to_words,
                                                                  params)
        if params['USE_DELTA']:
            self.delta = np.concatenate(self.delta, np.ones(self.key_array.shape[0] - self.delta.shape[0]))
        test_X, self.params['MAX_LENGTH'] = to_dense(test_X, replace = test_vocab_key)
        return cnn_eval.main(self.model, self.params, test_X, self.key_array,
                        measure, delta = self.delta)

    #softmax array
    def predict_prob(self, test_X, indices_to_words):
        return self.predict(test_X, indices_to_words, measure = 'scores')
