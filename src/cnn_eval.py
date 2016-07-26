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
                                    axis=0))
        test_X = test_X[1:]
    return all_x

def float_entropy(path, val_x, val_y, key_array, params):
    pred = evaluate(path, val_x, val_y, key_array, params, 'cross_entropy', np.zeros([0, key_array.shape[1]]))
    return np.mean(np.asarray(pred))

def evaluate(path, val_x, val_y, key_array, params, measure, new_key_embeds):
    with tf.Graph().as_default():
            cnn = CNN(params, key_array, batch_size=1, train=False)
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1,
                                          use_per_session_threads=True))
            saver = tf.train.Saver()
            saver.restore(sess, path)
            pred = []
            # try:
            #     print 'delta', cnn.W_delta.eval(session=sess)
            #     print cnn.stacked_W_delta.eval(session=sess)
            # except:
            #     pass
            # try:
            #     print 'wordembeds', cnn.weighted_word_embeddings.eval(session=sess)
            # except:
            #     pass
            # print 'l2 loss', tf.nn.l2_loss(cnn.weights[0]).eval(session=sess)
            while len(val_x) > 0:
                feed_dict = {cnn.input_x: np.expand_dims(val_x[0], axis=0),
                             cnn.input_y: np.expand_dims(val_y[0], axis=0),
                             cnn.dropout: 1.0,
                             cnn.word_embeddings_new: new_key_embeds}

                if measure == 'cross_entropy':
                    output = cnn.cross_entropy.eval(feed_dict=feed_dict, session=sess).tolist()
                elif measure == 'predict':
                    integer = cnn.predictions.eval(feed_dict=feed_dict, session=sess).tolist()
                    output = [0] * params['CLASSES']
                    output[integer] = 1
                else:
                    output = cnn.scores.eval(feed_dict=feed_dict, session=sess).tolist()[0]

                pred.append(output)
                val_x = val_x[1:]
                val_y = val_y[1:]
            return pred

def main(checkpoint, params, test_X, key_array, measure, new_key_embeds):
    test_Y_filler = [np.zeros(params['CLASSES'])] * len(test_X)
    pred = evaluate(checkpoint, test_X, test_Y_filler, key_array, params, measure, new_key_embeds)
    return pred

if __name__ == "__main__":
    main(checkpoint, params, test_X, key_array, measure)
