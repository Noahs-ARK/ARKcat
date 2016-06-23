
#other notes:
    # Adam does got lower softmax, by a couple %
    # dev softmax min is before dev accuracy max -- typically after epoch 2-4 (zero-based)
    # should we save the model with lowest cross entropy or highest accuracy?
    # learning rate should be higher with random init, lower when we update word vecs, lower for larger datasets

#minor changes:
    #shuffled batches each epoch

#todo:
    # debug changes in accuracy--is it the embeddings??
    # test with random addition of padding
    # what causes programs to stop??
    # clean up--more methods

#program expects:
    # flags: -a for Adagrad, -u for updating, -w for use word2vec, -t for use tfidf
        # default = Adam, no updating, random init w/o tfidf
    # argv[1] directory with train.data, dev.data, train.labels, dev.labels in SST format
    # argv[2] learning rate
    # argv[3] number of epochs
    # argv[4] tfidf ('True' or 'False')
    # argv[5] identifier tag (appended to filename to distinguish multiple runs)

#outputs in file named
    # directory, Optimizer name, number of epochs, identifier .txt
    # (commas only where necessary to distinguish numbers)
        # initial accuracy (train data)
        # training and dev accuracy at each epoch, dev softmax accuracy
import numpy as np
import tensorflow as tf
import random
import linecache
from text_cnn_methods_temp import *
from text_cnn_model_temp import CNN
import sys, argparse
import os
import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

#define hyperparameters
def define_globals(args):
    params = {'WORD_VECTOR_LENGTH' : 300,
        'FILTERS' : 100,
        'KERNEL_SIZES' : [3,4,5],
        'CLASSES' : 2,
        'MAX_LENGTH' : 59,

        'L2_NORM_CONSTRAINT' : 3.0,
        'TRAIN_DROPOUT' : 0.5,

        'BATCH_SIZE' : 50,
        'EPOCHS' : args.EPOCHS,
        'MAX_EPOCH_SIZE' : 10000,

        'Adagrad' : args.Adagrad,
        'LEARNING_RATE' : args.LEARNING_RATE,
        'USE_TFIDF' : args.tfidf,
        'USE_WORD2VEC' : args.word2vec,
        'USE_DELTA' : args.delta,
        'FLEX' : args.flex,
        'UPDATE_WORD_VECS' : args.update,
        'DIR' : args.path,
        'TRAIN_FILE_NAME' : 'train',
        'DEV_FILE_NAME' : 'dev',
        'WORD_VECS_FILE_NAME' : 'output.txt',
        'OUTPUT_FILE_NAME' : '' + str(args.path),
        'SST' : False,
        'ICMB' : False,
        'TREC' : False,
        'TEST' : False,

        #set by program-do not change!
        'epoch' : 1,
        'l2-loss' : tf.constant(0),
        #debug
        'key_errors' : [],
        'changes' : 0}
    return params

#change 'accuracy'
def evaluate(cnn, bundle, params, sess, cross_entropy=False):
        all_x, all_y, incomplete, extras, examples_total = bundle
        feed_dict = {cnn.input_x: all_x[0],
                     cnn.input_y: all_y[0],
                     cnn.dropout: 1.0}
        if cross_entropy == True:
            measure = cnn.log_loss
        else:
            measure = cnn.correct_prediction
        sum_correct = tf.reduce_sum(tf.cast(measure, dtype=tf.float32))
        examples_correct = 0
        if incomplete == False:
            while len(all_x) > 0:
                examples_correct += sum_correct.eval(feed_dict=feed_dict,
                                                     session = sess)
                all_x = all_x[1:]
                all_y = all_y[1:]
        else:
            while len(all_x) > 1:
                examples_correct += sum_correct.eval(feed_dict=feed_dict,
                                                     session = sess)
                all_x = all_x[1:]
                all_y = all_y[1:]
            final_batch = np.asarray(measure.eval(feed_dict=feed_dict,
                                                  session = sess))
            for i in range(0, params['BATCH_SIZE'] - extras):
                if final_batch[i] == True:
                    examples_correct += 1
        return float(examples_correct) / examples_total

def get_batches(params, train_eval_bundle, batches_bundle):
    if params['BATCH_SIZE'] == 1:
        return train_eval_bundle[:2]
    else:
        return scramble_batches(params, train_eval_bundle, batches_bundle)

#defunct; keeping for print statements
def regularize(output, weights, W_fc, biases, b_fc, params, sess):
    with sess.as_default():
        # if j == 0:
        #     l2_loss = tf.div(tf.sqrt(tf.nn.l2_loss(weights[0])), tf.convert_to_tensor(2.0)).eval()
        #     output.write('l2 loss is %g\n' %l2_loss)
        check_l2 = tf.reduce_sum(weights[0]).eval()
        for W in weights:
            W = tf.clip_by_average_norm(W, params['L2_NORM_CONSTRAINT'])
        for b in biases:
            b = tf.clip_by_average_norm(b, params['L2_NORM_CONSTRAINT'])
        W_fc = tf.clip_by_average_norm(W_fc, params['L2_NORM_CONSTRAINT'])
        b_fc = tf.clip_by_average_norm(b_fc, params['L2_NORM_CONSTRAINT'])
        if np.asscalar(check_l2) > np.asscalar(tf.reduce_sum(weights[0]).eval()):
            output.write('weights clipped\n')
    return weights, W_fc, biases, b_fc

def print_eval(output, name, bundle, params, cnn, sess):
    softmax = evaluate(cnn, bundle, params, sess, cross_entropy=True)
    accuracy = evaluate(cnn, bundle, params, sess)
    output.write(name + ' accuracy %g softmax %g \n'%(accuracy, softmax))
    return accuracy

def train_all(params, output, data):
    train_eval_bundle, dev_bundle, test_bundle, batches_bundle, key_array = data
    with tf.Graph().as_default():
        cnn = CNN(params, key_array)
        # var_update = optimizer.compute_gradients(tf.reduce_mean(cnn.log_loss))
        # train_op = optimizer.apply_gradients(var_update)
        train_step = cnn.optimizer.minimize(cnn.cross_entropy)
        saver = tf.train.Saver(tf.all_variables())
        #run session
        output.write( 'Initializing session...\n\n')
        sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2,
                          intra_op_parallelism_threads=3, use_per_session_threads=True))
        sess.run(tf.initialize_all_variables())
        output.write( 'Running session... setup time: %g\n'%(time.clock()))
        best_dev_accuracy = 0
        print_eval(output, 'initial', train_eval_bundle, params, cnn, sess)
        output.write('start time: ' + str(time.clock()) + '\n')
        time_index = time.clock()
        epoch_time = 0
        for i in range(params['EPOCHS']):
            params['epoch'] = i + 1
            batches_x, batches_y = get_batches(params, train_eval_bundle, batches_bundle)
            for j in range(len(batches_x)):
                feed_dict = {cnn.input_x: batches_x[j], cnn.input_y: batches_y[j],
                             cnn.dropout: params['TRAIN_DROPOUT']}
                train_step.run(feed_dict=feed_dict, session = sess)
                #apply l2 clipping to weights and biases
                check_l2 = tf.reduce_sum(cnn.weights[0]).eval(session = sess)
                cnn.clip_vars(params)
                if np.asscalar(check_l2) != np.asscalar(tf.reduce_sum(cnn.weights[0]).eval(session = sess)):
                    output.write('weights clipped\n')
            output.write("\nepoch %d\n"%params['epoch'])
            print_eval(output, 'training', train_eval_bundle, params, cnn, sess)
            dev_accuracy = print_eval(output, 'dev', dev_bundle, params, cnn, sess)
            if dev_accuracy > best_dev_accuracy:
                if params['TEST']:
                    checkpoint = saver.save(sess,
                        'text_cnn_run' + params['OUTPUT_FILE_NAME'],
                        global_step = params['epoch'])
                best_dev_accuracy = dev_accuracy
                # if dev_accuracy < best_dev_accuracy - .02:
                #     #early stop if accuracy drops significantly
                #     finish(output, best_dev_accuracy, epoch_time, params, cnn, test_bundle)
            output.write('epoch time : ' + str(time.clock() - time_index))
            epoch_time += time.clock() - time_index
            time_index = time.clock()
            output.write('. elapsed: ' + str(time.clock()) + '\n')
        finish(output, best_dev_accuracy, epoch_time, params, cnn, test_bundle)

def finish(output, best_dev_accuracy, epoch_time, params, cnn, test_bundle):
    output.write('avg time: %g\n' %(epoch_time/params['EPOCHS']))
    output.write('Max accuracy ' + str(best_dev_accuracy) + '\n')
    if params['TEST']:
        output.write('\nTesting:\n')
        saver.restore(sess, checkpoint)
        test_accuracy = evaluate(cnn, bundle, params, sess)
        output.write('Final test accuracy:')
            #%g' %g=test_accuracy)

def train_epoch(output, weights, W_fc, biases, b_fc, params, sess, train_eval_bundle, batches_bundle, x, y_, log_loss, correct_prediction, dropout):
    for j in range(len(batches_x)):
        batches_x, batches_y = get_batches(params, train_eval_bundle, batches_bundle)
        train_step.run(feed_dict={x: batches_x[j], y_: batches_y[j],
                            dropout: params['TRAIN_DROPOUT']}, session = sess)
        #apply l2 clipping to weights and biases
        weights, W_fc, biases, b_fc = regularize(output, weights, W_fc, biases, b_fc, params, sess)
    output.write("epoch %d"%params['epoch'])
    print_eval(output, 'training', x, y_, train_eval_bundle, params, log_loss, correct_prediction, dropout, sess)
    return print_eval(output, 'dev', x, y_, dev_bundle, params, log_loss, correct_prediction, dropout, sess)

def analyze_opts(args, params):
    if args.path == 'sst1':
        params['CLASSES'] = 5
        params['SST'] = True
    elif args.path == 'sst1':
        params['SST'] == True

    if params['Adagrad'] == True:
        params['OUTPUT_FILE_NAME'] += 'Adagrad'
    else:
        params['OUTPUT_FILE_NAME'] += 'Adam'
    params['OUTPUT_FILE_NAME'] += str(params['LEARNING_RATE'])

    if args.abbrev == True:
        params['KERNEL_SIZES'] = [3]
        params['FILTERS'] = 5
    if args.sgd == True:
        params['BATCH_SIZE'] = 1
        params['OUTPUT_FILE_NAME'] += 'sgd'
    return params

def get_data(params, output):
    train_x, train_y = get_all(params['DIR'], params['TRAIN_FILE_NAME'], params)

    dev_x, dev_y = get_all(params['DIR'], params['DEV_FILE_NAME'], params)
    if params['TEST']:
        test_x, test_y = get_all(params['DIR'], 'test', params)
    else:
        test_x, test_y = [],[]

    params['MAX_LENGTH'] = get_max_length(train_x + dev_x + test_x)
    vocab = find_vocab(train_x + dev_x + test_x, params)
    embed_keys, key_array = initialize_vocab(vocab, params)
    train_x, train_y = sort_examples_by_length(train_x, train_y)
    dev_x, dev_y = sort_examples_by_length(dev_x, dev_y)
    if params['TEST']:
        test_x, test_y = sort_examples_by_length(test_x, test_y)
        test_bundle = batch(test_x, test_y, params, embed_keys) + (len(test_y),)
    else:
        test_bundle = ()
    train_eval_bundle = batch(train_x, train_y, params, embed_keys) + (len(train_y),)
    dev_bundle = batch(dev_x, dev_y, params, embed_keys) + (len(dev_y),)
    batches_bundle = train_x, train_y, embed_keys
    output.write("Total vocab size: " + str(len(vocab))+ '\n')
    output.write('train set size: %d examples, %d batches per epoch\n'%(len(train_y), len(train_eval_bundle[0])))
    output.write("dev set size: " + str(len(dev_y))+ ' examples\n\n')
    return train_eval_bundle, dev_bundle, test_bundle, batches_bundle, key_array


def main(params, output):
    sys.stderr = output
    data = get_data(params, output)
    train_all(params, output, data)
    sys.stderr.close()
    sys.stderr = sys.__stderr__
    output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--Adagrad', action='store_true', default=False)
    parser.add_argument('-w', '--word2vec', action='store_true', default=False)
    parser.add_argument('-u', '--update', action='store_true', default=False)
    parser.add_argument('-d', '--delta', action='store_true', default=False)
    parser.add_argument('-e', '--test', action='store_true', default=False)
    parser.add_argument('-b', '--abbrev', action='store_true', default=False)
    parser.add_argument('-s', '--sgd', action='store_true', default=False)
    parser.add_argument('-t', '--tfidf', action='store_true', default=False)
    parser.add_argument('-f', '--flex', type=int, choices=xrange(1,10))
    parser.add_argument('path', type=str)
    parser.add_argument('LEARNING_RATE', type=float)
    parser.add_argument('EPOCHS', type=int)
    parser.add_argument('string')
    args = parser.parse_args()
    params = define_globals(args)
    params = analyze_opts(args, params)
    output = initial_print_statements(params, args)
    logging.basicConfig(filename=params['OUTPUT_FILE_NAME'], level=logging.DEBUG)
    try: main(params, output)
    except BaseException:
        logging.getLogger(__name__).exception("Program terminated")
        raise
