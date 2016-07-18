import numpy as np
import tensorflow as tf
import cnn_train
import cnn_eval
from cnn_methods import *
import os,sys
import scipy

#extant issues: 17 empty examles in train+val, then 17 each in train and val (?)
    #shows up as tf error at W_fc step
    #appears to be fixed, but need to remove debug code at some point
# dict fails
# vocab not in concordance


class Model_CNN:
    def __init__(self, params, n_labels, indices_to_words, model_dir):
        self.hp = params
        self.num_labels = n_labels
        self.indices_to_words = indices_to_words
        self.model_dir = model_dir

    def train(self, train_X, train_Y):
        # print 'train_X type:', type(train_X[0][0])
        print 'still needs debug!!!'
        self.params = {
                'model_num' : self.hp['model_num'],
                'FILTERS' : self.hp['filters'],
                'FILTERS' : 2,
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
                'KERNEL_SIZES' : [self.hp['kernel_size'],
                                  self.hp['kernel_size'] + self.hp['kernel_increment'],
                                  self.hp['kernel_size'] + 2 * self.hp['kernel_increment']],
                # 'KERNEL_SIZES' : self.hp['kernel_sizes']
                #'USE_DELTA' : self.hp['delta'],

                'WORD_VECTOR_LENGTH' : 300,
                'CLASSES' : self.num_labels,
                'EPOCHS' : 2,
        }
        self.vocab = get_vocab(self.indices_to_words)

        self.key_array = dict_to_array(self.vocab, self.params)
        # print self.vocab[:10]
        # self.key_array, self.vocab = dict_to_array2(self.indices_to_words, self.params)
        train_X, self.params['MAX_LENGTH'] = to_dense(train_X)
        train_Y = one_hot(train_Y, self.params['CLASSES'])
        if self.hp['flex']:
            self.params['FLEX'] = self.hp['flex_amt'] * self.params['MAX_LENGTH']
        else:
            self.params['FLEX'] = 0.0
        self.model = cnn_train.main(self.params, train_X, train_Y, self.key_array, self.model_dir)

    #one-hot array
    #don't need to save test key--regen each time for dev, etc
    def predict(self, test_X, indices_to_words=None, measure = 'predict'):
        print 'test_X type:', type(test_X[0])
        print 'debug types', type(test_X), type(indices_to_words)
        if 'numpy' not in str(type(test_X)):
            if indices_to_words is not None:
                test_vocab, test_key_array, test_vocab_key = process_test_vocab(self.vocab, self.key_array, indices_to_words, self.params)
                print 'made test key'
                print 'check test_vocab:', len(test_vocab)
                test_X, self.params['MAX_LENGTH'] = to_dense(test_X, test_key = test_vocab_key)
                print 'used key'

            else:
                test_vocab, test_key_array = self.vocab, self.key_array
                test_X, self.params['MAX_LENGTH'] = to_dense(test_X)
                print 'no key'
            for example in test_X[:10]:
                for word in example:
                    print test_vocab[word],
                print ''
        else:
            test_vocab, test_key_array = self.vocab, self.key_array

        #fails :(

        return cnn_eval.main(self.model, self.params, test_X, test_key_array,
                        measure)

    #softmax array
    def predict_prob(self, test_X, indices_to_words=None):
        return self.predict(test_X, indices_to_words=indices_to_words, measure = 'scores')
