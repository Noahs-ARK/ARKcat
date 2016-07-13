import tensorflow as tf
import sys, re
import random, math
import numpy as np
import os.path

#initializes weights, random with stddev of .1
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

#initializes biases, all at .1
def bias_variable(shape, name):
      initial = tf.zeros(shape=shape)
      return tf.Variable(initial, name=name)

def get_max_length(list_of_examples):
    max_length = 0
    for line in list_of_examples:
        max_length = max(max_length, len(line))
    return max_length

def get_max_numpy(list_of_examples):
    max_length = 0
    for line in list_of_examples:
        max_length = max(max_length, line.shape[0])
    return max_length

#takes a line of text, returns an array of strings where ecah string is a word
def tokenize(line):
   list_of_words = []
   word = ''
   for char in line:
      if char == ' ':
         list_of_words.append(word)
         word = ''
      else:
         word += char
   list_of_words.append(word.strip())
   return list_of_words

def flex(input_list, params):
    for example in input_list:
        if example.shape[0] + params['FLEX'] <= params['MAX_LENGTH']:
            #~30% chance of padding the left side
            if boolean_percent(15):
                example = insert_padding(example, params['FLEX'], True)
            elif boolean_percent(15):
                example = insert_padding(example, math.ceil(params['FLEX']/2.0), True)
            #~30% chance of padding the right
            if boolean_percent(15):
                example = insert_padding(example, params['FLEX'], False)
            elif boolean_percent(15):
                example = insert_padding(example, math.ceil(params['FLEX']/2.0), False)
    return input_list

def insert_padding(example, tokens_to_pad, left):
    if left:
        example = np.concatenate((np.zeros((tokens_to_pad)), example))
    else:
        example = np.concatenate((example, np.zeros((tokens_to_pad))))
    return example

def boolean_percent(percent):
    return random.randrange(100) < percent

#shuffle two numpy arrays in unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

#convert x to multidimensional python list if necessary
def scramble_batches(params, x, y):
    extras = len(x) % params['BATCH_SIZE']
    x, y = shuffle_in_unison(x, y)
    duplicates_x = []
    duplicates_y = []
    for i in range(extras):
        duplicates_x.append(x[i])
        duplicates_y.append(y[i])
    x.extend(duplicates_x)
    y.extend(duplicates_y)
    if params['FLEX'] > 0:
        x = flex(x, params)
    # if len(x) > params['MAX_EPOCH_SIZE']:
    #     extras = params['MAX_EPOCH_SIZE'] % params['BATCH_SIZE']
    #     x = x[:(params['MAX_EPOCH_SIZE']) - extras]
    #     y = y[:(params['MAX_EPOCH_SIZE']) - extras]
    #     incomplete = False
    x, y = sort_examples_by_length(x, y)
    batches_x, batches_y = [], []
    while len(y) >= params['BATCH_SIZE']:
        batches_x.append(pad_all(x[:params['BATCH_SIZE']], params))
        batches_y.append(np.asarray(y[:params['BATCH_SIZE']]))
        x = x[params['BATCH_SIZE']:]
        y = y[params['BATCH_SIZE']:]
    return batches_x, batches_y

def sort_examples_by_length(x, y):
    lengths = []
    for i in range(len(x)):
        lengths.append(len(x[i]))
    new_lengths = []
    new_x = []
    new_y = []
    for i in range(len(lengths)):
        for j in range(len(new_lengths)):
            if lengths[i] < new_lengths[j]:
                new_lengths.insert(j, lengths[i])
                new_x.insert(j, x[i])
                new_y.insert(j, y[i])
                break
        else:
            new_lengths.append(lengths[i])
            new_x.append(x[i])
            new_y.append(y[i])
    return new_x, new_y

#takes tokenized list_of_examples and pads all to the maximum length
def pad_all(list_of_examples, params):
    max_length = get_max_length(list_of_examples)
    for i in range(len(list_of_examples)):
        list_of_examples[i] = pad_one(list_of_examples[i], max_length, params)
    return list_of_examples

def one_hot(train_Y, CLASSES):
    one_hot = []
    for i in range(len(train_Y)):
       one_hot.append([0] * CLASSES)
       one_hot[i][train_Y[i]] = 1
       one_hot[i] = np.asarray(one_hot[i])
    return one_hot

#pads all sentences to same length
def pad_one(list_of_word_vecs, max_length, params):
    left = (max_length - len(list_of_word_vecs)) / 2
    right = left
    if (max_length - len(list_of_word_vecs)) % 2 != 0:
        right += 1
    return np.asarray(([0] * left) + list_of_word_vecs.tolist() + ([0] * right))

def to_dense(input_X):
    max_length = 0
    dense = []
    for example in input_X:
        example_transform = example[0].nonzero()[1]
        example_transform = example_transform.tolist()
        for word in example_transform:
            word += 1
        max_length = max(max_length, len(example_transform))
        dense.append(np.asarray(example_transform))
    return dense, max_length

def custom_loss(W, params):
        if params['REGULARIZER'] == 'l1':
            return tf.sqrt(tf.reduce_sum(tf.abs(W)))
        elif params['REGULARIZER'] == 'l2':
            return tf.sqrt(tf.scalar_mul(tf.constant(2.0), tf.nn.l2_loss(W)))
        else:
            return 0.0

def init_word_vecs(key_array, vocab, params):
    #with open('/Users/katya/repos/tensorflow/output-short.txt', 'r') as word2vec:
    with open('/home/katya/datasets/output.txt', 'r') as word2vec:
        word2vec.readline()
        for i in range(3000000):   #number of words in word2vec
            line = tokenize(word2vec.readline().strip())
            #turn into floats
            if line[0] in vocab:
                vector = []
                for word in line[1:]:
                    vector.append(float(word))
                if len(vector) != params['WORD_VECTOR_LENGTH']:
                    raise ValueError
                key_array[vocab.index(line[0])] = vector
    return key_array

def dict_to_array(d, params):
    vocab = []
    for word in d.itervalues():
        word = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", word)
        vocab.append(str(word))
    key_array = [[] for item in range(len(vocab))]
    #DEBUG: add filepath in user input
    if params['USE_WORD2VEC']:
        key_array = init_word_vecs(key_array, vocab, params)
    for i in range(len(key_array)):
        if key_array[i] == []:
            key_array[i] = np.random.uniform(-0.25,0.25,params['WORD_VECTOR_LENGTH'])
    key_array.insert(0, [0] * params['WORD_VECTOR_LENGTH'])
    return np.asarray(key_array), vocab

def separate_train_and_val(train_X, train_Y):
    shuffle_in_unison(train_X, train_Y)
    val_split = len(train_X)/10
    return train_X[val_split:], train_Y[val_split:], train_X[:val_split], train_Y[:val_split]

# def cnn_save_model()
# #see run.save_model
#     feature_list = result['model'].feats_and_params[0]['feats']
#     model_hyperparams = result['model'].feats_and_params[0]['params']
#     #STUPID FILENAMES TOO LONG
#     short_name = {'model_type':'mdl', 'regularizer':'rg', 'converg_tol':'cvrg','alpha':'alpha',
#                   'eta':'eta', 'gamma':'gamma', 'max_depth':'dpth', 'min_child_weight':'mn_wght',
#                   'max_delta_step':'mx_stp', 'subsample':'smpl', 'reg_strength':'rg_str',
#                   'num_round':'rnds', 'lambda':'lmbda', 'ngram_range':'ngrms', 'binary':'bnry',
#                   'use_idf':'idf', 'stop_words':'st_wrd', 'word_vector_init' : 'wv_init',
#                   'word_vector_update' : 'upd', 'delta': 'delta', 'flex': 'flex',
#                   'kernel_size_1':'ks1', 'kernel_size_2' :'ks2', 'kernel_size_3': 'ks3',
#                   'filters': 'fltrs', 'dropout': 'dropout', 'batch_size' : 'batch_size',
#                   'activation_fn': 'actvn_fn', 'regularizer':'rg', 'reg_strength':'rg_str',
#                   'learning_rate': 'learn_rt'}
#
#     # to save the model after each iteration
#     feature_string = ''
#     for feat, value in feature_list.items():
#         feature_string = feature_string + short_name[feat] + '=' + str(value) + ';'
#     for hparam in model_hyperparams:
#         cur_hparam = None
#         #DEBUGGING
#         if hparam == 'folds':
#             continue
#         if isinstance(model_hyperparams[hparam], float):
#             cur_hparam = str(round(model_hyperparams[hparam]*1000)/1000)
#         else:
#             cur_hparam = str(model_hyperparams[hparam])
#         feature_string = feature_string + short_name[hparam] + '=' + cur_hparam + ';'
#     feature_string = feature_string[:-1]


if __name__ == "__main__": main()
