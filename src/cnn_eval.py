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

def float_entropy(checkpoint, val_x, val_y, key_array, params):
    pred = evaluate(checkpoint, val_x, val_y, key_array, params, cross_entropy=True)
    return np.mean(pred)

#change 'accuracy'
def evaluate(checkpoint, val_x, val_y, key_array, params, cross_entropy, reinit_word_embeddings=False):
        print 'val sets', type(val_x), type(val_y)
        print 'val 1', type(val_x[0]), val_x[0], val_x[0].shape
        #cnn.input_x = tf.placeholder(tf.int32, [2, None], name='input_x')
        #cnn.input_y = tf.placeholder(tf.float32, [2, params['CLASSES']], name='input_y')
        with tf.Graph().as_default():
            cnn = CNN(params, key_array, batch_size=1)
            saver = tf.train.Saver(tf.all_variables())
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                                          intra_op_parallelism_threads=3,
                                          use_per_session_threads=True))
            saver.restore(sess, checkpoint)
            if reinit_word_embeddings:
                cnn.reinit_word_embeddings(val_x, params, sess)
            if cross_entropy:
                measure = cnn.cross_entropy
            else:
                measure = cnn.correct_prediction
            pred = []
            while len(val_x) > 0:
                feed_dict = {cnn.input_x: np.expand_dims(val_x[0], axis = 0),
                             cnn.input_y: np.expand_dims(val_y[0], axis = 0),
                             cnn.dropout: 1.0}
                pred.append(measure.eval(feed_dict=feed_dict, session = sess))
                val_x = val_x[1:]
                val_y = val_y[1:]
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
