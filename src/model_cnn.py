import numpy as np
import cnn_train
import cnn_eval
from cnn_methods import *
import scipy

#extant issues:
#test all hyperparams
#set up and test big cnn space
#clip_vars
class Model_CNN:
    def __init__(self, params, n_labels, indices_to_words, model_dir, word2vec_filename):
        self.hp = params
        self.num_labels = n_labels
        self.indices_to_words = fix_indices(indices_to_words)
        self.model_dir = model_dir
        self.word2vec_filename = word2vec_filename

    def train(self, train_X, train_Y):
        print 'fix later'
        self.params = {
                'FILTERS' : self.hp['filters'],
                # 'FILTERS' : 2,
                'ACTIVATION_FN' : self.hp['activation_fn'],
                'REGULARIZER' : self.hp['regularizer'],
                # 'REGULARIZER' : 'l2_clip',
                'REG_STRENGTH' : self.hp['reg_strength'],
                # 'REG_STRENGTH' : 2.0,
                'TRAIN_DROPOUT' : self.hp['dropout'],
                'BATCH_SIZE' : self.hp['batch_size'],
                'LEARNING_RATE' : self.hp['learning_rate'],
                'KERNEL_SIZES' : [],
                'USE_WORD2VEC' : self.hp['word_vector_init'],
                # 'USE_WORD2VEC' : False,
                'UPDATE_WORD_VECS' : self.hp['word_vector_update'],
                # 'UPDATE_WORD_VECS' : False,
                # 'USE_DELTA' : False,
                'USE_DELTA' : self.hp['delta'],

                'WORD_VECTOR_LENGTH' : 300,
                'CLASSES' : self.num_labels,
                'EPOCHS' : 5,
        }
        if self.params['REGULARIZER'] == 'l2':
            self.params['REG_STRENGTH'] = 10 ** self.params['REG_STRENGTH']
        for i in range(self.hp['kernel_num']):
            self.params['KERNEL_SIZES'].append(self.hp['kernel_size'] + i * self.hp['kernel_increment'])

        self.vocab = get_vocab(self.indices_to_words)
        self.key_array = dict_to_array(self.word2vec_filename, self.vocab, self.params)
        print 'shape', (train_X[0][0]).shape
        train_X, self.params['MAX_LENGTH'] = to_dense(train_X)
        print 'shape2', (train_X[0].shape)
        train_Y = one_hot(train_Y, self.params['CLASSES'])
        for example in train_X[:10]:
            for word in example:
                print self.vocab[word],
            print ''
        if self.hp['flex']:
            self.params['FLEX'] = int(self.hp['flex_amt'] * self.params['MAX_LENGTH'])
        else:
            self.params['FLEX'] = 0
        self.model = cnn_train.main(self.params, train_X, train_Y, self.key_array, self.model_dir)

    def predict(self, test_X, indices_to_words=None, measure='predict'):
        # print 'nested train_labels (called pred again)'
        # try:
        #     print test_X[0]
        #     try:
        #         print test_X[0][0]
        #         try:
        #             print test_X[0][0][0]
        #             try:
        #                 print test_X[0][0][0][0]
        #             except:
        #                 print '3'
        #                 pass
        #         except:
        #             print '2'
        #             pass
        #     except:
        #         print '1'
        #         pass
        # except:
        #     print 'none'
        #     pass
        if 'numpy' not in str(type(test_X)):
            #if called on dev or test
            if indices_to_words is not None:
                # print vocab_debug(test_X[0:10], indices_to_words)

                placeholder, test_key_array, test_vocab_key = process_test_vocab(self.word2vec_filename, self.vocab, indices_to_words, self.params)
                test_X, self.params['MAX_LENGTH'] = to_dense(test_X, test_key=test_vocab_key)
                return cnn_eval.main(self.model, self.params, test_X,
                                     np.concatenate((self.key_array, test_key_array), axis=0),
                                     measure)

            #called on train
            else:
                test_X, self.params['MAX_LENGTH'] = to_dense(test_X)


        return cnn_eval.main(self.model, self.params, test_X, self.key_array,
                                 measure)


    #softmax array
    def predict_prob(self, test_X, indices_to_words=None):
        return self.predict(test_X, indices_to_words=indices_to_words, measure='scores')
