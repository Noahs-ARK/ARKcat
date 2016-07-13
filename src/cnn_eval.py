#publish code as scikit module
import numpy as np
import tensorflow as tf
from cnn_methods import *
from cnn_class import *
import sys, os

def test_batch(test_X, params, embed_keys):
    all_x = []
    while len(output_list) > 0:
        all_x.append(np.expand_dims(sub_indices_one(input_list[0], embed_keys),
                                    axis = 0))
        test_X = test_X[1:]
    return all_x

def float_entropy(path, checkpoint, val_x, val_y, key_array, params):
    pred = evaluate(path, checkpoint, val_x, val_y, key_array, params, 'cross_entropy')
    return np.mean(np.asarray(pred))

def evaluate(path, checkpoint, val_x, val_y, key_array, params, measure):
    # print 'got here'
    with tf.Graph().as_default():
    #     with sess.as_default():
            cnn = CNN(params, key_array, batch_size=1)
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1,
                                          use_per_session_threads=True))
            for var in tf.all_variables():
                print var.name
            saver = tf.train.Saver()
            saver.restore(sess, path)
            if measure == 'cross_entropy':
                evaluation = cnn.cross_entropy
            elif measure == 'predict':
                evaluation = cnn.predictions
            else:
                evaluation = cnn.scores
            pred = []
            while len(val_x) > 0:
                feed_dict = {cnn.input_x: np.expand_dims(val_x[0], axis = 0),
                             cnn.input_y: np.expand_dims(val_y[0], axis = 0),
                             cnn.dropout: 1.0}
                pred.append(evaluation.eval(feed_dict=feed_dict, session = sess))
                val_x = val_x[1:]
                val_y = val_y[1:]
            return pred

def main(checkpoint, params, test_X, key_array, measure):
    test_Y_filler = [np.zeros(params['CLASSES'])] * len(test_X)
    pred = evaluate(checkpoint, test_X, test_Y_filler, key_array, params, measure=measure)
    print 'pred0', pred[0]
    return pred

if __name__ == "__main__":
    main(checkpoint, params, test_X, key_array, measure)
