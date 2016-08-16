import numpy as np
import tensorflow as tf
from cnn_methods import *
from cnn_class import CNN
import cnn_eval
import time, resource
import os, sys #debug

#remove any old chkpt files
def remove_chkpt_files(epoch, model_dir):
    for past_epoch in range(epoch):
        file_path = model_dir + 'temp_cnn_eval_epoch%i' %(past_epoch)
        try:
            os.remove(file_path)
            os.remove(file_path + '.meta')
        except (UnboundLocalError, OSError):
            pass

def epoch_write_statements(timelog, init_time, epoch):
    timelog.write('\n\nepoch %i initial time %g' %(epoch, time.clock()))
    timelog.write('\n epoch time %i\navg time: %g' %((time.time() - init_time), (time.time() - init_time)/ (epoch + 1)))
    timelog.write('\nCPU usage: %g'
                %(resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                resource.getrusage(resource.RUSAGE_SELF).ru_stime))
    timelog.write('\nmemory usage: %g' %(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

def return_current_state(saver, sess, model_dir, epoch):
    path_final = saver.save(sess, model_dir + 'cnn_final%i' %epoch)
    return path_final, cnn.word_embeddings.eval(session=sess)

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

def initial_prints(timelog, saver, sess, model_dir, val_X, val_Y, key_array, params):
    timelog.write('\n\n\nNew Model:')
    init_time = time.time()
    timelog.write(str(init_time))

    path = saver.save(sess, model_dir + 'temp_cnn_eval_epoch%i' %0)
    best_dev_loss = cnn_eval.float_entropy(path, val_X, val_Y, key_array, params)

    timelog.write( '\ndebug dev loss %g' %best_dev_loss)
    timelog.write('\n%g'%time.clock())
    return best_dev_loss, init_time

def new_best_model(timelog, saver, sess, cnn, path, dev_loss):
    timelog.write('\nnew best model')
    path_final = saver.save(sess, path)
    return dev_loss, cnn.word_embeddings.eval(session=sess)

def set_up_model(params, key_array):
    cnn = CNN(params, key_array)
    loss = cnn.cross_entropy
    loss += tf.mul(tf.constant(params['REG_STRENGTH']), cnn.reg_loss)
    train_step = cnn.optimizer.minimize(loss)
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=1, use_per_session_threads=True))
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
    return cnn, loss, train_step, sess, saver

def epoch_train(train_X, train_Y, params, cnn, sess, train_step):
    batches_x, batches_y = scramble_batches(params, train_X, train_Y)

    for j in range(len(batches_x)):
        feed_dict = {cnn.input_x: batches_x[j], cnn.input_y: batches_y[j],
                     cnn.dropout: params['TRAIN_DROPOUT'],
                     cnn.word_embeddings_new: np.zeros([0, key_array.shape[1]])}
        train_step.run(feed_dict=feed_dict, session=sess)
        #apply l2 clipping to weights and biases
        if params['REGULARIZER'] == 'l2_clip':
            cnn = clip_tensors(j, len(batches_x), cnn, sess, params)
    return cnn

def main(params, input_X, input_Y, key_array, model_dir):
    train_X, train_Y, val_X, val_Y = separate_train_and_val(input_X, input_Y)
    # debugging_saved_model = False

    with tf.Graph().as_default():
        with open(model_dir + 'train_log', 'a') as timelog:
            #make init method
            cnn, loss, train_step, sess, saver = set_up_model(params, key_array)
            best_dev_loss, init_time = initial_prints(timelog, saver, sess, model_dir, val_X, val_Y, key_array, params)

            for epoch in range(params['EPOCHS']):
                cnn = epoch_train(train_X, train_Y, params, cnn, sess, train_step)
                epoch_write_statements(timelog, init_time, epoch)

                path = saver.save(sess, model_dir + 'temp_cnn_eval_epoch%i' %(epoch))
                dev_loss = cnn_eval.float_entropy(path, val_X, val_Y, key_array, params)
                timelog.write('\ndev accuracy: %g'%dev_loss)
                if dev_loss < best_dev_loss:
                    best_dev_loss, word_embeddings = new_best_model(timelog, saver, sess, cnn,
                                                    model_dir + 'cnn_final%i' %epoch, dev_loss)
                elif dev_loss > best_dev_loss + .05:
                    remove_chkpt_files(epoch, model_dir)
                    #early stop if accuracy drops significantly
                    try:
                        return path_final, word_embeddings
                    except (UnboundLocalError, ValueError): #path_final does not exist because initial dev accuracy highest
                        return return_current_state(saver, sess, model_dir, epoch)

            remove_chkpt_files(epoch, model_dir)
            try:
                return path_final, word_embeddings
            except (UnboundLocalError, ValueError): #path_final does not exist because initial dev accuracy highest
                return return_current_state(saver, sess, model_dir, epoch)

if __name__ == "__main__":
    main()

    #tri sgd
