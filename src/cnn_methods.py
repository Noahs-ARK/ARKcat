import random, math
import numpy as np

#assorted snippets of code used by model_cnn, cnn_train, and cnn_eval

#max length of example in minibatch
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

#randomly adds zero padding to some examples to increase minibatch variety
def flex(input_list, params):
    for example in input_list:
        if example.shape[0] + params['FLEX'] <= params['MAX_LENGTH']:
            #~30% chance of padding the left side
            if boolean_percent(15):
                example = insert_padding(example, params['FLEX'], True)
            elif boolean_percent(15):
                example = insert_padding(example, int(math.ceil(params['FLEX']/2.0)), True)
            #~30% chance of padding the right
            if boolean_percent(15):
                example = insert_padding(example, params['FLEX'], False)
            elif boolean_percent(15):
                example = insert_padding(example, int(math.ceil(params['FLEX']/2.0)), False)
    return input_list

#inserts zero padding for flex--note this is on a np array which has already been processed for feeding into tf
def insert_padding(example, tokens_to_pad, left):
    if left:
        example = np.concatenate((np.zeros((tokens_to_pad)), example))
    else:
        example = np.concatenate((example, np.zeros((tokens_to_pad))))
    return example

#returns a boolean true in percent of cases
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
#MAX_EPOCH_SIZE might be useful when dealing with very large datasets
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

# sorts examples in input_x by length
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

# converts list of ints into list of one_hot vectors (np arrays)
#for purposes of calculating cross_entropy loss
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

#returns nonzero entries in input_X as np array
def to_dense(input_X, test_key = None):
    max_length = 0
    dense = []
    for example in input_X:
        example_transform = example[0].nonzero()[1]
        example_transform = example_transform.tolist()
        for word in example_transform:
            if test_key is not None:
                temp_test_key = test_key
                word = test_key[word]
            word += 1
        max_length = max(max_length, len(example_transform))
        dense.append(np.asarray(example_transform))
    return dense, max_length


#gets word vecs from word2vec_filename. those not found will be initialized later
def init_word_vecs(word2vec_filename, key_array, vocab, params):
    #with open('/Users/katya/repos/tensorflow/output-short.txt', 'r') as word2vec:
    with open(word2vec_filename, 'r') as word2vec:
        word2vec.readline()
        for i in range(3000000):   #number of words in word2vec
            line = tokenize(word2vec.readline().strip())
            #check to see if it contains nonAscii (which would break the if statement)
            try:
                line[0].decode('ascii')
            except UnicodeDecodeError:
                pass
            #turns word vectors into floats and appends to key array
            else:
                if line[0] in vocab:
                    vector = []
                    for word in line[1:]:
                        vector.append(float(word))
                    if len(vector) != params['WORD_VECTOR_LENGTH']:
                        raise ValueError
                    key_array[vocab.index(line[0])] = vector
    return key_array

#returns an array of new word vectors for that vocab,
#and dict to link indices in test_X with indices in key_array
def process_test_vocab(word2vec_filename, vocab, new_vocab_key, params):
    add_vocab_list = []
    for word in new_vocab_key.itervalues():
        if word not in vocab:
            add_vocab_list.append(word)
    new_key_array = dict_to_array(word2vec_filename, add_vocab_list, params, train=False)
    all_vocab = vocab + add_vocab_list
    for key in new_vocab_key.iterkeys():
        new_vocab_key[key] = all_vocab.index(new_vocab_key[key])
    return all_vocab, new_key_array, new_vocab_key
    #     add_vocab_list = []
#     for word in new_vocab_key.itervalues():
#         if word not in vocab:
#             add_vocab_list.append(word)
#     new_key_array = dict_to_array(word2vec_filename, add_vocab_list, params, train=False)
#     all_vocab = vocab + add_vocab_list
#     for key in new_vocab_key.iterkeys():
#         new_vocab_key[key] = all_vocab.index(new_vocab_key[key])
#     try:
#         for example in test_X[:10]:
#             for word in example:
#                 try:
#                     print new_vocab_key[word],
#                     print all_vocab[new_vocab_key[word]],
#                 except IndexError:
#                     print 'IndexError'
#             print ''
#     except KeyError:
#         print 'KeyError'
#         print 'word', word
#         print 'new_vocab_key.popitem', new_vocab_key.popitem()
#         print ''
#         test_X = rm_empty_dim(test_X)
#         print 'rm dim'
#         for example in test_X:
#             for word in example:
#                 try:
#                     print new_vocab_key[word],
#                     print all_vocab[new_vocab_key[word]],
#                 except IndexError:
#                     print 'IndexError'
#             print ''
#
#     print 'len vocab', len(all_vocab)
#     print 'len array', new_key_array.shape
#
#     return new_key_array, new_vocab_key
#
#loads word vectors
def dict_to_array(word2vec_filename, vocab, params, train=True):
    key_array = [[] for item in range(len(vocab))]
    if params['USE_WORD2VEC']:
        key_array = init_word_vecs(word2vec_filename, key_array, vocab, params)
    for i in range(len(key_array)):
        if key_array[i] == []:
            key_array[i] = np.random.uniform(-0.25,0.25,params['WORD_VECTOR_LENGTH'])
    if train:
        key_array.insert(0, [0] * params['WORD_VECTOR_LENGTH'])
    return np.asarray(key_array)

#saves vocab from TfidfVectorizer in list. indices in self.vocab will match those in key_array
def get_vocab(indices_to_words):
    vocab = [None] * len(indices_to_words)
    for key in indices_to_words:
        vocab[key] = indices_to_words[key]
    return vocab

def separate_train_and_val(train_X, train_Y):
    shuffle_in_unison(train_X, train_Y)
    val_split = len(train_X)/10
    return train_X[val_split:], train_Y[val_split:], train_X[:val_split], train_Y[:val_split]

#transforms indices_to_words to needed form
def fix_indices(indices_to_words):
    for key in indices_to_words:
        key += 1
    #dummy key so that len() matches
    indices_to_words[0] = None
    return indices_to_words

def rm_empty_dim(test_X):
    for i in range(len(test_X)):
        test_X[i] = test_X[i][0]

def vocab_debug(debug_X, indices_to_words):
    if 'numpy' not in str(type(debug_X[0])):
        debug2_X, l = to_dense(debug_X)
    for example in debug2_X[:10]:
        for word in example:
            try:
                print indices_to_words[word],
            except IndexError:
                print 'IndexError'
        print ''

if __name__ == "__main__": main()
