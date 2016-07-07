import tensorflow as tf
import sys, re
import random
import numpy as np
import math
import os.path

def initial_print_statements(params, args):
    params['OUTPUT_FILE_NAME'] += ',%i'%params['EPOCHS']
    if params['USE_TFIDF']:
        params['OUTPUT_FILE_NAME'] += 'tfidf'
    params['OUTPUT_FILE_NAME'] += ','
    if params['USE_WORD2VEC']:
        params['OUTPUT_FILE_NAME'] += 'word2vec'
    else:
        params['OUTPUT_FILE_NAME'] += 'randinit'
    params['OUTPUT_FILE_NAME'] += ','
    if params['UPDATE_WORD_VECS']:
        params['OUTPUT_FILE_NAME'] += 'upd'
    if params['USE_DELTA']:
        params['OUTPUT_FILE_NAME'] += 'delta'
    params['OUTPUT_FILE_NAME'] += args.string + '.txt'
    output = open(params['OUTPUT_FILE_NAME'], 'a', 0)
    if params['Adagrad']:
        output.write("Running Adagrad on %s with a learning rate of " %args.path)
    else:
        output.write("Running Adam on %s with a learning rate of " %args.path)
    output.write('%g and %i epochs\n'%(params['LEARNING_RATE'], params['EPOCHS']))
    output.write('using batch size %i' %(params['BATCH_SIZE']))
    if params['USE_TFIDF']:
        output.write(', tfidf, ')
    else:
        output.write(', ')
    if params['USE_WORD2VEC']:
        output.write('word2vec, ')
    else:
        output.write('rand init, ')
    if params['UPDATE_WORD_VECS']:
        output.write('updating.\n')
    else:
        output.write('not updating.\n')
    return output

#breaks when BATCH_SIZE = 1
def batch(input_list, output_list, params, embed_keys):
    all_x, all_y = [], []
    if params['BATCH_SIZE'] == 1:
        while len(output_list) > 0:
            # print 'remaining lengeth', len(output_list)
            # print 'batches', len(all_y)
            # print input_list[0]
            all_x.append(np.expand_dims(sub_indices_one(input_list[0], embed_keys), axis = 0))
            all_y.append(np.reshape(np.asarray(output_list[0]), (1, 2)))
            input_list = input_list[1:]
            output_list = output_list[1:]
        # print 'consecutive ex', all_x[1], all_x[2]
        return all_x, all_y, False, 0
    while len(output_list) >= params['BATCH_SIZE']:
        # print 'start'
        # print sub_indices(pad_all(input_list[:params['BATCH_SIZE']]), embed_keys)
        all_x.append(sub_indices(pad_all(input_list[:params['BATCH_SIZE']]), embed_keys))
        all_y.append(np.asarray(output_list[:params['BATCH_SIZE']]))
        input_list = input_list[params['BATCH_SIZE']:]
        output_list = output_list[params['BATCH_SIZE']:]
    if len(output_list) > 0:
        extras = params['BATCH_SIZE'] - len(output_list)
        input_list = sub_indices(pad_all(input_list), embed_keys)
        all_y.append(np.concatenate((np.asarray(output_list), np.zeros((extras, params['CLASSES']))), axis = 0))
        zeroes = np.full((extras, input_list.shape[1]), embed_keys['<PAD>'], dtype=int)
        all_x.append(np.concatenate((input_list, zeroes), axis = 0))
        return all_x, all_y, True, extras
    else:
        return all_x, all_y, False, 0

#index and loop through same batches again
def get_batch(batches_x, batches_y, index, params):
    cur_batch_x = batches_x[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    cur_batch_y = batches_y[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    # if params['USE_TFIDF']:
    #     cur_batch_tfidf = batches_tfidf[index*params['BATCH_SIZE']:(index+1)*params['BATCH_SIZE'],:]
    return cur_batch_x, cur_batch_y

#initializes weights, random with stddev of .1
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#initializes biases, all at .1
def bias_variable(shape):
      initial = tf.zeros(shape=shape)
      return tf.Variable(initial)

def one_hot(category, CLASSES):
    one_hot = [0] * CLASSES
    one_hot[category] = 1
    return np.asarray(one_hot)

def sub_indices(input_x, embed_keys):
    example_list = []
    for sentence in input_x:
        example_list.append(sub_indices_one(sentence, embed_keys))
    return np.asarray(example_list)

def sub_indices_one(sentence, embed_keys):
    list_of_indices = []
    for token in sentence:
        list_of_indices.append(embed_keys[token])
    return list_of_indices


#get all examples from a file and return np arrays w/input and output
def get_all(directory, file_name, params):
    input_file = open(os.path.expanduser("~") + '/convnets/tensorflow/' + os.path.join(directory, file_name) + '.data', 'r')
    output_file = open(os.path.expanduser("~") + '/convnets/tensorflow/' + os.path.join(directory, file_name) + '.labels', 'r')
    # input_file = open(os.path.join(os.path.expanduser("~") + '/repos/tensorflow/' + file_name) + '.data', 'r')
    # output_file = open(os.path.join(os.path.expanduser("~") + '/repos/tensorflow/' + file_name) + '.labels', 'r')
    input_list = []
    output_list = []
    for line in input_file:
        input_list.append(tokenize(clean_str(line, params)))
    for line in output_file:
        output_list.append(one_hot(int(line.rstrip()), params['CLASSES']))
    return input_list, output_list

#shuffle two numpy arrays in unison
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    # np.random.set_state(rng_state)
    # np.random.shuffle(c)
    return a, b

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

#imported from
def clean_str(string, params):
    if params['SST'] == True:
        """
        Tokenization/string cleaning for the SST dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()
    if params['ICMB'] == True:
        string = string.replace('<br /><br />', ' ')
        string = re.sub(r"-", " - ", string)
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if params['TREC'] else string.strip().lower()

#create a vocabulary list from a file
def find_vocab(list_of_sentences, params, vocab=None):
    list_of_words = []
    for sentence in list_of_sentences:
        list_of_words.extend(sentence)
    if vocab is None:
        vocab = []
    list_of_words.insert(0, '<PAD>')
    for word in list_of_words:
        if word not in vocab:
            vocab.append(word)
    return vocab

#initialize dict of vocabulary with word2vec or random numbers
def initialize_vocab(vocab, params, embed_keys = {}):
    key_list = []
    if params['USE_WORD2VEC']:
        word2vec = open(params['WORD_VECS_FILE_NAME'], 'r')
        word2vec.readline()
        for i in range(3000000):   #number of words in word2vec
            line = tokenize(word2vec.readline().strip())
            #turn into floats
            if line[0] in vocab and line[0] not in embed_keys:
                vector = []
                for word in line[1:]:
                    vector.append(float(word))
                # print len(vector), vector
                if len(vector) != params['WORD_VECTOR_LENGTH']:
                    raise ValueError
                key_list.append(vector)
                embed_keys[line[0]] = len(embed_keys)
                vocab.remove(line[0])
        word2vec.close()
    for word in vocab:
        if word == '<PAD>':
            key_list.append(np.zeros([params['WORD_VECTOR_LENGTH']]))
        else:
            key_list.append(np.random.uniform(-0.25,0.25,params['WORD_VECTOR_LENGTH']))
        embed_keys[word] = len(embed_keys)
    return embed_keys, np.asarray(key_list)

def custom_loss(W, params):
        if params['REGULARIZER'] == 'l1':
            return tf.sqrt(tf.reduce_sum(tf.abs(W)))
        elif params['REGULARIZER'] == 'l2':
            return tf.sqrt(tf.scalar_mul(tf.constant(2), tf.nn.l2_loss(W)))
        else:
            return 0

if __name__ == "__main__": main()
