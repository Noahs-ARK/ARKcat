import numpy as np
import tensorflow as tf
from cnn_methods import *
from cnn_class import CNN
import cnn_eval
import time, resource
import os, sys

#batching methods
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

#takes tokenized list_of_examples and pads all to the maximum length
def pad_all(list_of_examples, params):
    max_length = get_max_length(list_of_examples)
    for i in range(len(list_of_examples)):
        list_of_examples[i] = pad_one(list_of_examples[i], max_length, params)
    return list_of_examples

#pads all sentences to same length
def pad_one(list_of_word_vecs, max_length, params):
    left = (max_length - len(list_of_word_vecs)) / 2
    right = left
    if (max_length - len(list_of_word_vecs)) % 2 != 0:
        right += 1
    return np.asarray(([0] * left) + list_of_word_vecs.tolist() + ([0] * right))

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
    x, y = sort_examples_by_length(x, y)
    batches_x, batches_y = [], []
    while len(y) >= params['BATCH_SIZE']:
        batches_x.append(pad_all(x[:params['BATCH_SIZE']], params))
        batches_y.append(np.asarray(y[:params['BATCH_SIZE']]))
        x = x[params['BATCH_SIZE']:]
        y = y[params['BATCH_SIZE']:]
    return batches_x, batches_y

def separate_train_and_val(train_X, train_Y):
    shuffle_in_unison(train_X, train_Y)
    val_split = len(train_X)/10
    return train_X[val_split:], train_Y[val_split:], train_X[:val_split], train_Y[:val_split]

#remove any existing old chkpt files, ignore nonexistent ones
#DEBUGGING: remove timelog from here
def remove_chkpt_files(epoch, model_dir):
    for past_epoch in range(epoch):
        file_path = model_dir + 'temp_cnn_eval_epoch%i' %(past_epoch)
        if os.path.isfile(file_path) and os.path.isfile(file_path + '.meta'):
            os.remove(file_path)
            os.remove(file_path + '.meta')

def epoch_write_statements(timelog, init_time, epoch):
    timelog.write('\n\n')
    timelog.write('epoch %i start time %g\n' %(epoch, time.clock()))
    timelog.write('total time spent since epoch 1: %i\n' %((time.time() - init_time)))
    timelog.write('avg time per epoch: %g\n' %((time.time() - init_time)/ (epoch + 1)))
    timelog.write('CPU usage: %g\n'
                %(resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                resource.getrusage(resource.RUSAGE_SELF).ru_stime))
    timelog.write('memory usage: %g' %(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

#debug method--writes all files in model_dir to file timelog
def print_debug_paths(model_dir, timelog):
    for dirname, dirnames, filenames in os.walk(model_dir):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            timelog.write(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            timelog.write(os.path.join(dirname, filename))

#dependency of print_clips method
def l2_loss_float(W):
    return tf.sqrt(tf.scalar_mul(tf.convert_to_tensor(2.0), tf.nn.l2_loss(W)))

#debug method--prints the l2 norm of weights for purpose of checking l2 clipping
def print_clips(cnn, sess, params):
    check_weights = tf.reduce_sum(cnn.weights[0]).eval(session=sess)
    check_biases = tf.reduce_sum(cnn.biases[0]).eval(session=sess)
    check_Wfc = tf.reduce_sum(cnn.W_fc).eval(session=sess)
    check_bfc = tf.reduce_sum(cnn.b_fc).eval(session=sess)
    cnn.clip_vars(params)
    weights_2 = tf.reduce_sum(cnn.weights[0]).eval(session=sess)
    biases_2 = tf.reduce_sum(cnn.biases[0]).eval(session=sess)
    Wfc_2 = tf.reduce_sum(cnn.W_fc).eval(session=sess)
    bfc_2 = tf.reduce_sum(cnn.b_fc).eval(session=sess)
    if np.array_equal(check_weights, weights_2):
        print 'clipped'
    elif np.array_equal(check_biases, biases_2):
        print 'clipped'
    elif np.array_equal(check_Wfc, Wfc_2):
        print 'clipped'
    elif np.array_equal(check_bfc, bfc_2):
        print 'clipped'
    else:
        print 'no clip. means:'
    print l2_loss_float(cnn.weights[0]).eval(session=sess)
    print l2_loss_float(cnn.biases[0]).eval(session=sess)
    print l2_loss_float(cnn.W_fc).eval(session=sess)
    print l2_loss_float(cnn.b_fc).eval(session=sess)
    return cnn

def clip_tensors(j, length, cnn, sess, params):
    if j == (length - 2):
        cnn = print_clips(cnn, sess, params)
    else:
        cnn.clip_vars(params)
    return cnn

def initial_prints(timelog, saver, sess, model_dir, val_X, val_Y, word_vec_array, params):
    timelog.write('\n\n\nNew Model:')
    init_time = time.time()
    timelog.write(str(init_time))
    path = saver.save(sess, model_dir + 'temp_cnn_eval_epoch%i' %0)

    best_dev_loss = cnn_eval.float_entropy(path, val_X, val_Y, word_vec_array, params)

    timelog.write( '\ndebug dev loss %g' %best_dev_loss)
    timelog.write('\n%g'%time.clock())
    return best_dev_loss, init_time

def set_up_model(params, word_vec_array):
    cnn = CNN(params, word_vec_array)
    train_step = cnn.train_op # optimizer.minimize(loss)
    with cnn.graph.as_default():
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
    return cnn, cnn.loss, train_step, saver

def epoch_train(train_X, train_Y, word_vec_array, params, cnn, train_step, timelog):
    batches_x, batches_y = scramble_batches(params, train_X, train_Y)

    total_train_step_time = 0
    total_clip_time = 0
    for j in range(len(batches_x)):
        feed_dict = {cnn.input_x: batches_x[j], cnn.input_y: batches_y[j],
                     cnn.dropout: params['TRAIN_DROPOUT'],
                     cnn.word_embeddings_new: np.zeros([0, word_vec_array.shape[1]])}
        start_train_step_time = time.time()
        train_step.run(feed_dict=feed_dict, session=cnn.sess)
        total_train_step_time = total_train_step_time + (time.time() - start_train_step_time)

        #apply l2 clipping to weights and biases
        #DEBUGGING
        if params['REGULARIZER'] == 'l2_clip' and False:
            start_clip_time = time.time()
            cnn.clip_vars(params)
            total_clip_time = total_clip_time + (time.time() - start_clip_time)
    timelog.write('\nthe amount of time the training step takes: %g' %(total_train_step_time))
    timelog.write('\nthe amount of time clipping_vars takes: %g' %(total_clip_time))
    return cnn

def print_if_file_doesnt_exist(timelog, file_path, cur_loc):
    if not os.path.isfile(file_path):
        timelog.write("found that ")

def train(params, input_X, input_Y, word_vec_array, model_dir):
    
    train_X, train_Y, val_X, val_Y = separate_train_and_val(input_X, input_Y)
    path_final, word_embeddings = None, None
    with open(model_dir + 'train_log', 'a') as timelog:
        # with tf.Graph().as_default(), tf.Session() as sess:
        cnn, loss, train_step, saver = set_up_model(params, word_vec_array)
        best_dev_loss, init_time = initial_prints(timelog, saver, cnn.sess, model_dir, val_X, 
                                                  val_Y, word_vec_array, params)

        for epoch in range(params['EPOCHS']):
            cnn = epoch_train(train_X, train_Y, word_vec_array, params, cnn, train_step, timelog)
            epoch_write_statements(timelog, init_time, epoch)

            start_save_time = time.time()
            path = saver.save(cnn.sess, model_dir + 'temp_cnn_eval_epoch%i' %(epoch))
            total_save_time = time.time() - start_save_time
            timelog.write("\nthe amount of time it takes to save the model at epoch " + 
                          str(epoch) + ": %g" %(total_save_time))

            float_entropy_init_time = time.time()
            dev_loss = cnn_eval.float_entropy(path, val_X, val_Y, word_vec_array, params)
            float_entropy_time = time.time() - float_entropy_init_time
            timelog.write('\ndev cross entropy, best cross entropy: %g, %g  (it took %g '
                          %(dev_loss, best_dev_loss, float_entropy_time) + 'seconds to compute)')
            timelog.flush()


            start_save_best_time = time.time()
            if dev_loss < best_dev_loss:
                timelog.write('\nnew best model, saving model in ' + (model_dir + 'cnn_final%i' %epoch))
                best_dev_loss = dev_loss
                path_final = saver.save(cnn.sess, model_dir + 'cnn_final%i' %epoch)
                word_embeddings = cnn.word_embeddings.eval(session=cnn.sess)
            #early stop if accuracy drops significantly
            elif dev_loss > best_dev_loss + .02:
                break
            total_save_best_time = time.time() - start_save_best_time
            timelog.write('\nhow long saving the best model at the end takes: %g' %(total_save_best_time))

        remove_chkpt_files(epoch, model_dir)
        
        if (path_final is None or word_embeddings is None):
            timelog.write('failure to train, returning current state')
            path_final = saver.save(cnn.sess, model_dir + 'cnn_final%i' %epoch)
            word_embeddings = cnn.word_embeddings.eval(session=cnn.sess)

    return path_final, word_embeddings
