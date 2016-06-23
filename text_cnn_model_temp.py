import numpy as np
import tensorflow as tf
from text_cnn_methods_temp import *

class CNN(object):
    def __init__(self, params, key_array):
        self.input_x = tf.placeholder(tf.int32, [params['BATCH_SIZE'], None])
        self.input_y = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['CLASSES']])
        self.dropout = tf.placeholder(tf.float32)
        self.embeddings = tf.Variable(tf.convert_to_tensor(key_array, dtype = tf.float32),
                                      trainable = params['UPDATE_WORD_VECS'])

        self.embedding_output = embed_layer(self.input_x, params, key_array, embeddings)

        #init lists for convolutional layer
        slices = []
        self.weights = []
        self.biases = []
        #loop over KERNEL_SIZES, each time initializing a slice
        for kernel_size in params['KERNEL_SIZES']:
            W = weight_variable([kernel_size, 1, params['WORD_VECTOR_LENGTH'], params['FILTERS']])
            b = bias_variable([params['FILTERS']])
            #convolve: each neuron iterates by 1 filter, 1 word
            conv = tf.nn.conv2d(self.embedding_output, W, strides=[1, 1, 1, 1], padding="SAME")
            #apply bias and relu
            relu = tf.nn.relu(tf.nn.bias_add(conv, b))
            #max pool; each neuron sees 1 filter and returns max over a sentence

            pooled = tf.nn.max_pool(relu, ksize=[1, params['MAX_LENGTH'], 1, 1],
                strides=[1, params['MAX_LENGTH'], 1, 1], padding='SAME')
            slices.append(pooled)
            self.weights.append(W)
            self.biases.append(b)
        self.h_pool = tf.concat(3, slices)
        self.h_pool_drop = tf.nn.dropout(self.h_pool, self.dropout)
        self.h_pool_flat = tf.reshape(self.h_pool_drop, [params['BATCH_SIZE'], -1])
        #fully connected softmax layer
        self.W_fc = weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'],
                                params['CLASSES']])
        self.b_fc = bias_variable([params['CLASSES']])
        self.scores = tf.nn.softmax(tf.nn.xw_plus_b(self.h_pool_flat, self.W_fc, self.b_fc))
        self.predictions = tf.argmax(self.scores, 1)
        #define error for training steps
        self.log_loss = -tf.reduce_sum(self.input_y * tf.log(self.scores),
                                       reduction_indices=[1])
        self.cross_entropy = tf.reduce_mean(self.log_loss)
        #define accuracy for evaluation
        self.correct_prediction = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
        if params['Adagrad']:
            self.optimizer = tf.train.AdagradOptimizer(params['LEARNING_RATE'])
        else:
            self.optimizer = tf.train.AdamOptimizer(params['LEARNING_RATE'])

    def reinit_word_embeddings(new_key_array, params):
        with sess.as_default():
            embeddings_array = np.concat((self.embeddings.eval(), new_key_array))
            self.embeddings = tf.Variable(tf.convert_to_tensor(embeddings_array,
                                          dtype = tf.float32),
                                          trainable = params['UPDATE_WORD_VECS'])

    def clip_vars(self, params):
        for W in self.weights:
            W = tf.clip_by_average_norm(W, params['L2_NORM_CONSTRAINT'])
        for b in self.biases:
            b = tf.clip_by_average_norm(b, params['L2_NORM_CONSTRAINT'])
        self.W_fc = tf.clip_by_average_norm(self.W_fc, params['L2_NORM_CONSTRAINT'])
        self.b_fc = tf.clip_by_average_norm(self.b_fc, params['L2_NORM_CONSTRAINT'])
