import tensorflow as tf
import sys, re
import random, math
import numpy as np
import os.path

#initializes weights, random with stddev of .1
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#initializes biases, all at .1
def bias_variable(shape):
      initial = tf.zeros(shape=shape)
      return tf.Variable(initial)

def get_max_length(list_of_examples):
    max_length = 0
    for line in list_of_examples:
        max_length = max(max_length, len(line))
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

def to_dense(input_X, replace = None):
    max_length = 0
    for i in range(len(input_X)):
        input_X[i] = input_X[i][0].nonzero()[1]
        input_X[i] = input_X[i].tolist()
        for word in input_X[i]:
            if replace is not None:
                word = replace[word]
            word += 1
        max_length = max(max_length, len(input_X[i]))
        input_X[i] = np.asarray(input_X[i])
    return input_X, max_length

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

def update_vocab(key_array, vocab, indices_to_words, params):
    map_test_vocab = {}
    new_vocab = []
    words_to_indices = {process(v): k for k, v in indices_to_words.items()}
    for word in words_to_indices.iterkeys():
        if word in vocab:
            map_test_vocab[words_to_indices[word]] = vocab.find(word)
        else:
            new_vocab.append(words_to_indices[word])
            map_test_vocab[words_to_indices[word]] = len(new_vocab) + len(vocab)
    print map_test_vocab
    for word in new_vocab:
        new_key_array = dict_to_array(new_vocab, params)
    return np.concatenate(key_array, new_key_array), map_test_vocab, vocab.extend(new_vocab)

def process(vocab):
    result = []
    for word in vocab:
        word = re.sub(r"[^A-Za-z0-9(),!?\'\`]", "", word)
        result.append(str(word))
    return result

def dict_to_array(vocab, params):
    key_array = [[] for item in range(len(vocab))]
    #DEBUG: add filepath in user input
    if params['USE_WORD2VEC']:
        key_array = init_word_vecs(vocab, key_array)
    for i in range(len(key_array)):
        if key_array[i] == []:
            key_array[i] = np.random.uniform(-0.25,0.25,params['WORD_VECTOR_LENGTH'])
    key_array.insert(0, [0] * params['WORD_VECTOR_LENGTH'])
    return np.asarray(key_array)

def separate_train_and_val(train_X, train_Y):
    shuffle_in_unison(train_X, train_Y)
    val_split = len(train_X)/10
    return train_X[val_split:], train_Y[val_split:], train_X[:val_split], train_Y[:val_split]

if __name__ == "__main__": main()
