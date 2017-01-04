import numpy as npp
import cnn_train
import cnn_eval
from cnn_methods import *
import scipy
import tensorflow as tf
import time
import cProfile, pstats, sys

# converts list of ints into list of one_hot vectors (np arrays)
#for purposes of calculating cross_entropy loss
def one_hot(train_Y, CLASSES):
    one_hot = []
    for i in range(len(train_Y)):
       one_hot.append([0] * CLASSES)
       one_hot[i][train_Y[i]] = 1
       one_hot[i] = np.asarray(one_hot[i])
    return one_hot

#returns nonzero entries in input_X as np array
#if all_word_to_index != None, it remaps the elements of input_X to use the new vocab
#passed in all_word_to_index
def to_dense(input_X, all_word_to_index=None, test_index_to_word=None):
    max_length = 0
    dense = []
    for example in input_X:
        example_transform = example[0].nonzero()[1]
        example_transform = example_transform.tolist()
        for i in range(len(example_transform)):
            if all_word_to_index is not None:
                example_transform[i] = all_word_to_index[test_index_to_word[example_transform[i]]]
        max_length = max(max_length, len(example_transform))
        dense.append(np.asarray(example_transform))
    return dense, max_length

#what we have: 
#word_vecs:= word->vec
#word_to_index:= word->index
#what we want:
#array, where at index i we have an array of word vector with word_to_index index i
def make_array_of_vecs(word_to_index, word_vecs, params, train=True):
    word_vec_array = [None] * len(word_to_index)
    if params['USE_WORD2VEC']:
        for word in word_to_index:
            if word in word_vecs:
                word_vec_array[word_to_index[word]] = word_vecs[word]
    for i in range(len(word_vec_array)):
        if word_vec_array[i] == None:
            word_vec_array[i] = np.random.uniform(-0.25,0.25, params['WORD_VECTOR_LENGTH'])
    return np.asarray(word_vec_array)
    
#saves word_to_index from TfidfVectorizer in list. indices in self.word_to_index will match those in word_vec_array
def get_word_to_index(index_to_word):
    word_to_index = {}
    for key in index_to_word:
        if index_to_word[key] in word_to_index:
            #DEBUGGING
            #This shouldn't happen, it only happens when the vocab has duplicates
            print("in model_cnn.get_word_to_index()")
            print("found word that's already in word_to_index, trying to add it again! bad news bears:",
                  (key, index_to_word[key]))
            sys.stdout.flush()
        word_to_index[index_to_word[key]] = key
    return word_to_index
            
class Model_CNN:
    def __init__(self, params, n_labels, index_to_word, model_dir, train_word_vecs):
        self.train_counter = 0
        self.hp = params
        self.num_labels = n_labels
        self.word_to_index = get_word_to_index(index_to_word)
        self.model_dir = model_dir
        self.train_word_vecs = train_word_vecs
        self.set_params()

    #grayed out options force model to stay away from expensive options, speeding DEBUGGING
    #you can also reduce the number of epochs
    def set_params(self):
        self.params = {
                'FILTERS' : self.hp['filters'],
                'ACTIVATION_FN' : self.hp['activation_fn'],
                'REGULARIZER' : self.hp['regularizer'],
                'REG_STRENGTH' : self.hp['reg_strength'],
                'TRAIN_DROPOUT' : self.hp['dropout'],
                'BATCH_SIZE' : self.hp['batch_size'],
                'LEARNING_RATE' : self.hp['learning_rate'],
                'KERNEL_SIZES' : [],
                'USE_WORD2VEC' : self.hp['word_vector_init'],
                'UPDATE_WORD_VECS' : self.hp['word_vector_update'],
                'USE_DELTA' : self.hp['delta'],

                'WORD_VECTOR_LENGTH' : 300,
                'CLASSES' : self.num_labels,
                'EPOCHS' : 15
                # debug regularization:
                # 'REGULARIZER' : 'l2',
                # 'REG_STRENGTH' : -8.0,
                # 'REGULARIZER': 'l2_clip',
                # 'REG_STRENGTH': 2.0,
                # smaller: for general debugging:
                # 'FILTERS' : 2,
                # 'KERNEL_SIZES' : [3],
                # 'USE_WORD2VEC' : False,
                # 'UPDATE_WORD_VECS' : False,
                # 'USE_DELTA' : False,
                # 'EPOCHS' : 2
        }
        if self.params['REGULARIZER'] == 'l2':
            self.params['REG_STRENGTH'] = 10 ** self.params['REG_STRENGTH']
        for i in range(self.hp['kernel_num']):
            self.params['KERNEL_SIZES'].append(self.hp['kernel_size'] + i * self.hp['kernel_increment'])

    def train(self, train_X, train_Y):
        self.word_vec_array = make_array_of_vecs(self.word_to_index, self.train_word_vecs, 
                                                 self.params, train=True)
        train_X, self.params['MAX_LENGTH'] = to_dense(train_X)
        train_Y = one_hot(train_Y, self.params['CLASSES'])
        if self.hp['flex']:
            self.params['FLEX'] = int(self.hp['flex_amt'] * self.params['MAX_LENGTH'])
        else:
            self.params['FLEX'] = 0
        self.best_epoch_path, self.word_vec_array = cnn_train.train(self.params,
                                train_X, train_Y, self.word_vec_array, self.model_dir)        
        

    #helper method
    #takes two maps: all_word_to_index, and test_word_to_vec
    #returns array of word vecs, in the order suggested by 
    #takes train and test word_to_indexularies and test_word_to_vec
    #returns new word_to_index which has all words from train with original indices,
    #and the new words from test with incides greater than the original train indices,
    #and returns array of new word vectors which is in the same order as the new indices
    def get_all_word_to_index(self, test_index_to_word, train_word_to_index, test_word_to_vec):
        test_word_vec_array = []
        all_word_to_index = {}
        word_to_index_size = max(train_word_to_index.values())
        for word in train_word_to_index:
            all_word_to_index[word] = train_word_to_index[word]
        for test_word in test_index_to_word.values():
            if test_word not in train_word_to_index:

                word_to_index_size += 1
                all_word_to_index[test_word] = word_to_index_size
                if test_word in test_word_to_vec:
                    test_word_vec_array.append(test_word_to_vec[test_word])
                else:
                    test_word_vec_array.append(np.random.uniform(-0.25,0.25,
                                                                 self.params['WORD_VECTOR_LENGTH']))
        return all_word_to_index, test_word_vec_array

    def predict(self, test_X, test_index_to_word=None, test_word_to_vec=None, measure='predict'):
        if 'numpy' not in str(type(test_X)):
            #if called on dev or test
            if test_index_to_word is not None:
                all_word_to_index, test_word_vec_array = self.get_all_word_to_index(test_index_to_word, 
                                                                self.word_to_index, test_word_to_vec)
                test_X, self.params['MAX_LENGTH'] = to_dense(test_X, 
                                                             all_word_to_index=all_word_to_index,
                                                             test_index_to_word=test_index_to_word)
                return cnn_eval.dev_or_test_acc(self.best_epoch_path, self.params, test_X,
                                     self.word_vec_array, measure, test_word_vec_array)
                
                
            #called on train
            else:
                test_X, self.params['MAX_LENGTH'] = to_dense(test_X)

        return cnn_eval.dev_or_test_acc(self.best_epoch_path, self.params, test_X, self.word_vec_array,
                                 measure, np.zeros([0, self.word_vec_array.shape[1]]))

    #softmax array
    def predict_prob(self, test_X, test_index_to_word=None, test_word_to_vec=None):
        return self.predict(test_X, test_index_to_word=test_index_to_word, 
                            test_word_to_vec=test_word_to_vec, measure='scores')
