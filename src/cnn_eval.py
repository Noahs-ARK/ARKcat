import numpy as np
import tensorflow as tf
from text_cnn_methods_temp import *
from text_cnn_model_temp import *
import sys, os

#examples are going to be scrambled
def get_test_data(test_X, vocab, embed_keys, key_array, params):
    vocab = initialize_vocab(test_X, vocab=vocab)
    embed_keys, new_key_array = initialize_vocab(vocab, params,
                                                 embed_keys=embed_keys)
    test_bundle = test_batch(test_X, params, embed_keys)
    return test_bundle, vocab, embed_keys, new_key_array

def test_batch(test_X, params, embed_keys):
    all_x = []
    while len(output_list) > 0:
        all_x.append(np.expand_dims(sub_indices_one(input_list[0], embed_keys),
                                    axis = 0))
        test_X = test_X[1:]
    return all_x

#change 'accuracy'
def evaluate(cnn, val_x, val_y, params, sess, cross_entropy=False):
        print 'val sets', type(val_x), type(val_y)
        print 'val 1', type(val_x[0]), val_x[0], val_x[0].shape
        #cnn.input_x = tf.placeholder(tf.int32, [2, None], name='input_x')
        #cnn.input_y = tf.placeholder(tf.float32, [2, params['CLASSES']], name='input_y')
        print cnn.__dict__
        print cnn.input_x
        feed_dict = {cnn.input_x: np.stack([val_x[0]] * params['BATCH_SIZE']),
                     cnn.input_y: np.asarray([val_y[0]] * params['BATCH_SIZE']),
                     cnn.dropout: 1.0}
        print 'check!', feed_dict[cnn.input_x], feed_dict[cnn.input_y]
        print type(feed_dict[cnn.input_x])
        if cross_entropy == True:
            measure = cnn.cross_entropy
        else:
            measure = cnn.correct_prediction
        pred = []
        while len(val_x) > 0:
            pred.append(measure.eval(feed_dict=feed_dict, session = sess))
            val_x = val_x[1:]
            val_y = val_y[1:]
            print 'test', feed_dict[cnn.input_x]
        return np.asarray(pred)

def main(checkpoint, params, test_bundle, key_array, prob):
    with tf.Graph().as_default():
        cnn = CNN(params, key_array)
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                                  intra_op_parallelism_threads=3,
                                  use_per_session_threads=True))
        saver.restore(sess, checkpoint)
        cnn.reinit_word_embeddings(new_key_array, params, sess)
        return evaluate(cnn, test_bundle, params, sess, cross_entropy = prob)

if __name__ == "__main__":
    main(checkpoint, params, test_bundle, key_array)
