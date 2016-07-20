import numpy as np
import tensorflow as tf
import cnn_train
import cnn_eval
from cnn_methods import *
import os,sys
import scipy

#old issues: 17 empty examles in train+val, then 17 each in train and val (?)
    #shows up as tf error at W_fc step
    #appears to be fixed, but need to remove debug code at some point
# dict fails
# vocab not in concordance

#extant issues:
#test all hyperparams
#set up and test big cnn space
#why does cnn kernel size "change"?

class Model_CNN:
    def __init__(self, params, n_labels, indices_to_words, model_dir, word2vec_filename):
        self.hp = params
        self.num_labels = n_labels
        self.indices_to_words = indices_to_words
        self.model_dir = model_dir
        self.word2vec_filename = word2vec_filename

    def train(self, train_X, train_Y):
        print 'fix later'
        self.params = {
                # 'FILTERS' : self.hp['filters'],
                'FILTERS' : 2,
                'ACTIVATION_FN' : self.hp['activation_fn'],
                'REGULARIZER' : self.hp['regularizer'],
                'REG_STRENGTH' : self.hp['reg_strength'],
                'TRAIN_DROPOUT' : self.hp['dropout'],
                'BATCH_SIZE' : self.hp['batch_size'],
                'LEARNING_RATE' : self.hp['learning_rate'],
                'KERNEL_SIZES' : [],
                #'USE_WORD2VEC' : self.hp['use_word2vec'],
                'USE_WORD2VEC' : False,
                #'UPDATE_WORD_VECS' : self.hp['word_vector_update'],
                'UPDATE_WORD_VECS' : False,
                'USE_DELTA' : False,
                #'USE_DELTA' : self.hp['delta'],

                'WORD_VECTOR_LENGTH' : 300,
                'CLASSES' : self.num_labels,
                'EPOCHS' : 1,
        }
        if self.params['REGULARIZER'] == 'l2':
            self.params['REG_STRENGTH'] = 10 ** self.params['REG_STRENGTH']
        # for i in range(self.hp['kernel_num']):
        for i in range(1):
            self.params['KERNEL_SIZES'].append(self.hp['kernel_size'] + i * self.hp['kernel_increment'])

        self.vocab = get_vocab(self.indices_to_words)
        self.key_array = dict_to_array(self.word2vec_filename, self.vocab, self.params)

        train_X, self.params['MAX_LENGTH'] = to_dense(train_X)
        train_Y = one_hot(train_Y, self.params['CLASSES'])

        if self.hp['flex']:
            self.params['FLEX'] = int(self.hp['flex_amt'] * self.params['MAX_LENGTH'])
        else:

            self.params['FLEX'] = 0
        print
        self.model = cnn_train.main(self.params, train_X, train_Y, self.key_array, self.model_dir)

    #one-hot array
    #don't need to save test key--regen each time for dev, etc
    def predict(self, test_X, indices_to_words=None, measure='predict'):
        if 'numpy' not in str(type(test_X)):
            if indices_to_words is not None:
                test_vocab, test_key_array, test_vocab_key = process_test_vocab(self.word2vec_filename, self.vocab, indices_to_words, self.params)
                # print 'made test key'
                # print 'check test_vocab:', len(self.vocab + test_vocab)
                test_X, self.params['MAX_LENGTH'] = to_dense(test_X, test_key=test_vocab_key)
                # print 'used key'

                # print 'test_X:', test_X

                for example in test_X[:10]:
                    for word in example:
                        try:
                            print test_vocab[word],
                        except IndexError:
                            print 'IndexError'
                    print ''

                return cnn_eval.main(self.model, self.params, test_X, np.concatenate((self.key_array, test_key_array), axis=0),
                                measure)

            else:
                test_X, self.params['MAX_LENGTH'] = to_dense(test_X)
                print 'no key'
                for example in test_X[:10]:
                    for word in example:
                        print self.vocab[word],
                    print ''
        else:
            for example in test_X[:10]:
                for word in example:
                    print self.vocab[word],
                print ''

        return cnn_eval.main(self.model, self.params, test_X, self.key_array,
                        measure)


    #softmax array
    def predict_prob(self, test_X, indices_to_words=None):
        return self.predict(test_X, indices_to_words=indices_to_words, measure='scores')
