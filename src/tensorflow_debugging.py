import tensorflow as tf
from cnn_class import *
import numpy as np



def evaluate(checkpoint):
    with tf.Graph().as_default():
        #first = tf.Variable(tf.constant(0), name='first')
        cnn = CNN(params, key_array)
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, checkpoint)
        for var in tf.trainable_variables():
            print(var)
            print(var.name)

#first = tf.Variable(tf.constant(0), name='first')
global params, key_array
params = {
    'model_num' : 0,
    'FLEX' : 0,
    'FILTERS' : 10,
    'ACTIVATION_FN' : "iden",
    'REGULARIZER' : None,
    'REG_STRENGTH' : 0.0,
    'TRAIN_DROPOUT' : .4,
    'BATCH_SIZE' : 2,
    'LEARNING_RATE' : .1,
    'USE_WORD2VEC' : False,
    'UPDATE_WORD_VECS' : False,
    'KERNEL_SIZES' : [5,
                      5,
                      5],
    'USE_DELTA' : False,
    'WORD_VECTOR_LENGTH' : 300,
    'CLASSES' : 2,
    'EPOCHS' : 15,
    'epoch' : 1,
    'l2-loss' : tf.constant(0),
    'MAX_LENGTH' : 20
}

key_array = np.zeros(shape=(10,300))

cnn = CNN(params, key_array)
loss = cnn.cross_entropy
loss += tf.mul(tf.constant(params['REG_STRENGTH']), cnn.reg_loss)
train_step = cnn.optimizer.minimize(loss)

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(init_op)

path = saver.save(sess, '/home/jesse/scratch/python/saved_vars/tmp.vars')
print(path)

saver.restore(sess, path)

evaluate(path)
