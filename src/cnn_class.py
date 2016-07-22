import numpy as np
import tensorflow as tf

#problem: need to discard cnn btwn models
#else gets error when shape of vars doesn't match :()
class CNN:
    def __init__(self, params, key_array, batch_size=None):
        if batch_size == None:
            batch_size = params['BATCH_SIZE']
        self.input_x = tf.placeholder(tf.int32, [batch_size, None])#, name='input_x')
        self.input_y = tf.placeholder(tf.float32, [batch_size, params['CLASSES']])#, name='input_y')
        self.dropout = tf.placeholder(tf.float32)#, name='dropout')
        word_embeddings = tf.Variable(tf.convert_to_tensor(key_array, dtype=tf.float32), validate_shape=False,
                                      trainable = params['UPDATE_WORD_VECS'], name='word_embeddings')
        if params['USE_DELTA']:
            W_delta = tf.Variable(tf.ones, key_array.shape[0], validate_shape=False, name='W_delta')
            weighted_word_embeddings = tf.matmul(word_embeddings, W_delta)
            embedding_output = tf.nn.embedding_lookup(weighted_word_embeddings, self.input_x)
        else:
            embedding_output = tf.nn.embedding_lookup(word_embeddings, self.input_x)
        embedding_output = tf.expand_dims(embedding_output, 2)

        slices = []
        self.weights = []
        self.biases = []
        name_counter = 1

        #loop over KERNEL_SIZES, each time initializing a slice
        for kernel_size in params['KERNEL_SIZES']:
            W = self.weight_variable([kernel_size, 1, params['WORD_VECTOR_LENGTH'], params['FILTERS']], 'W_%i' %name_counter)
            b = self.bias_variable([params['FILTERS']], 'b_%i' %name_counter)
            name_counter += 1
            conv = tf.nn.conv2d(embedding_output, W, strides=[1, 1, 1, 1], padding="SAME")
            if params['ACTIVATION_FN'] == 'relu':
                activ = tf.nn.relu(tf.nn.bias_add(conv, b))
            elif params['ACTIVATION_FN'] == 'elu':
                activ = tf.nn.elu(tf.nn.bias_add(conv, b))
            # elif params['ACTIVATION_FN'] == 'tanh':
            #     activ = tf.nn.tanh(tf.nn.bias_add(conv, b))
            else:
                activ = conv
            pooled = tf.nn.max_pool(activ, ksize=[1, params['MAX_LENGTH'], 1, 1],
                strides=[1, params['MAX_LENGTH'], 1, 1], padding='SAME') #name='max_pool')

            slices.append(pooled)
            self.weights.append(W)
            self.biases.append(b)

        self.h_pool = tf.concat(3, slices)
        self.h_pool_drop = tf.nn.dropout(self.h_pool, self.dropout)
        self.h_pool_flat = tf.reshape(self.h_pool_drop, [batch_size, -1])
        #fully connected softmax layer
        self.W_fc = self.weight_variable([len(params['KERNEL_SIZES']) * params['FILTERS'],
                                params['CLASSES']], 'W_fc')
        self.b_fc = self.bias_variable([params['CLASSES']], 'b_fc')
        self.scores = tf.nn.softmax(tf.nn.xw_plus_b(self.h_pool_flat, self.W_fc, self.b_fc))
        self.predictions = tf.argmax(self.scores, 1)
        #define error for training steps
        self.cross_entropy = -tf.reduce_sum(self.input_y * tf.log(self.scores),
                                       reduction_indices=[1])
        #define accuracy for evaluation
        self.correct_prediction = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
        self.reg_loss = tf.constant(0.0)
        if params['UPDATE_WORD_VECS']:
            self.reg_loss += self.custom_loss(word_embeddings, params)
        if params['USE_DELTA']:
            self.reg_loss += self.custom_loss(W_delta, params)
        for W in self.weights:
            self.reg_loss += self.custom_loss(W, params)
        for b in self.biases:
            self.reg_loss += self.custom_loss(b, params)
        self.reg_loss += self.custom_loss(self.W_fc, params)
        self.reg_loss += self.custom_loss(self.b_fc, params)
        self.optimizer = tf.train.AdamOptimizer(params['LEARNING_RATE'])

    #calculates l1 and l2 losses for feeding to optimizer
    def custom_loss(self, W, params):
        if params['REGULARIZER'] == 'l1':
            return tf.sqrt(tf.reduce_sum(tf.abs(W)))
        elif params['REGULARIZER'] == 'l2':
            return tf.sqrt(tf.scalar_mul(tf.constant(2.0), tf.nn.l2_loss(W)))
        else:
            return 0.0

    #still not working :(
    def clip_vars(self, params):
        for W in self.weights:
            W = tf.clip_by_average_norm(W, params['REG_STRENGTH'])
        for b in self.biases:
            b = tf.clip_by_average_norm(b, params['REG_STRENGTH'])
        self.W_fc = tf.clip_by_average_norm(self.W_fc, params['REG_STRENGTH'])
        self.b_fc = tf.clip_by_average_norm(self.b_fc, params['REG_STRENGTH'])

    #initializes weights, random with stddev of .1
    def weight_variable(self, shape, name):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name=name)

    #init biases with stddev or .05
    def bias_variable(self, shape, name):
          initial = tf.truncated_normal(shape, stddev=0.05)
          return tf.Variable(initial, name=name)
