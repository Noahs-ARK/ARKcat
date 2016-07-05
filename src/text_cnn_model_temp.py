import numpy as np
import tensorflow as tf
from text_cnn_methods_temp import *

class CNN:
    def __init__(self, params, vocab):

        self.input_x = tf.placeholder(tf.int32, [params['BATCH_SIZE'], None])
        self.input_y = tf.placeholder(tf.float32, [params['BATCH_SIZE'], params['CLASSES']])
        self.dropout = tf.placeholder(tf.float32)

        word_embeddings = tf.Variable(tf.convert_to_tensor(key_array, dtype = tf.float32),
                                      trainable = params['UPDATE_WORD_VECS'])
        if params['USE_DELTA']:
            W_delta = tf.Variable(tf.ones, key_array.shape[0])
            weighted_word_embeddings = tf.matmul(word_embeddings, W_delta)
            embedding_output = tf.nn.embedding_lookup(weighted_word_embeddings, x)
        else:
            embedding_output = tf.nn.embedding_lookup(word_embeddings, x)

        #init lists for convolutional layer
        slices = []
        weights = []
        biases = []
        #loop over KERNEL_SIZES, each time initializing a slice
        for kernel_size in params['KERNEL_SIZES']:
            W = weight_variable([kernel_size, 1, params['WORD_VECTOR_LENGTH'], params['FILTERS']])
            b = bias_variable([params['FILTERS']])
            #convolve: each neuron iterates by 1 filter, 1 word
            conv = tf.nn.conv2d(embedding_output, W, strides=[1, 1, 1, 1], padding="SAME")
            #apply bias and activation fn
            if params['ACTIVATION_FN'] == 'relu':
                activ = tf.nn.relu(tf.nn.bias_add(conv, b))
            elif params['ACTIVATION_FN'] == 'elu':
                activ = tf.nn.elu(tf.nn.bias_add(conv, b))
            elif params['ACTIVATION_FN'] == 'tanh':
                activ = tf.nn.tanh(tf.nn.bias_add(conv, b))
            else:
                activ = conv
            #max pool; each neuron sees 1 filter and returns max over a sentence

            pooled = tf.nn.max_pool(activ, ksize=[1, params['MAX_LENGTH'], 1, 1],
                strides=[1, params['MAX_LENGTH'], 1, 1], padding='SAME')
            slices.append(pooled)
            weights.append(W)
            biases.append(b)
        self.h_pool = tf.concat(3, slices)
        self.h_pool_drop = tf.nn.dropout(self.h_pool, self.dropout)
        self.h_pool_flat = tf.reshape(self.h_pool_drop, [params['BATCH_SIZE'], -1])
        #fully connected softmax layer
        W_fc = weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'],
                                params['CLASSES']])
        b_fc = bias_variable([params['CLASSES']])
        self.scores = tf.nn.softmax(tf.nn.xw_plus_b(self.h_pool_flat, W_fc, b_fc))
        self.predictions = tf.argmax(self.scores, 1)
        #define error for training steps
        self.cross_entropy = -tf.reduce_sum(self.input_y * tf.log(self.scores),
                                       reduction_indices=[1])
        #define accuracy for evaluation
        self.correct_prediction = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
        self.reg_loss = tf.constant(0)
        if params['UPDATE_WORD_VECS']:
            self.reg_loss += custom_loss(word_embeddings, params)
        if params['USE_DELTA']:
            self.reg_loss += custom_loss(W_delta, params)
        for W in weights:
            self.reg_loss += custom_loss(W, params)
        for b in biases:
            self.reg_loss += custom_loss(b, params)
        self.reg_loss += custom_loss(W_fc, params)
        self.reg_loss += custom_loss(b_fc, params)
        if params['Adagrad']:
            self.optimizer = tf.train.AdagradOptimizer(params['LEARNING_RATE'])
        else:
            self.optimizer = tf.train.AdamOptimizer(params['LEARNING_RATE'])

    # def reinit_word_embeddings(new_key_array, params, sess):
    #     with sess.as_default():
    #         embeddings_array = np.concat((self.embeddings.eval(), new_key_array))
    #         self.embeddings = tf.Variable(tf.convert_to_tensor(embeddings_array,
    #                                       dtype = tf.float32),
    #                                       trainable = params['UPDATE_WORD_VECS'])

    def custom_loss(W, params):
        if params['REGULARIZER'] == 'l1':
            return tf.sqrt(tf.reduce_sum(tf.abs(W)))
        elif params['REGULARIZER'] == 'l2':
            return tf.sqrt(tf.scalar_mul(tf.constant(2), tf.nn.l2_loss(W)))
        else:
            return 0

    #needs debug
    def clip_vars(self, params):
        for W in self.weights:
            W = tf.clip_by_average_norm(W, params['L2_NORM_CONSTRAINT'])
        for b in self.biases:
            b = tf.clip_by_average_norm(b, params['L2_NORM_CONSTRAINT'])
        self.W_fc = tf.clip_by_average_norm(self.W_fc, params['L2_NORM_CONSTRAINT'])
        self.b_fc = tf.clip_by_average_norm(self.b_fc, params['L2_NORM_CONSTRAINT'])
