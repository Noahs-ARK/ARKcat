import numpy as np
import tensorflow as tf
from cnn_methods import *
from cnn_class import CNN
import cnn_eval
import time, resource
import os

#only training one epoch??

def l2_loss_float(W):
    return tf.sqrt(tf.scalar_mul(tf.convert_to_tensor(2.0), tf.nn.l2_loss(W)))

def remove_chkpt_files(file_path):
    os.remove(file_path)
    os.remove(file_path + '.meta')


def main(params, input_X, input_Y, key_array, model_dir, train_counter):
    os.makedirs(model_dir + params['MODEL_NUM'])
    train_X, train_Y, val_X, val_Y = separate_train_and_val(input_X, input_Y)

    with tf.Graph().as_default():
        with open(model_dir + 'train_log', 'a') as timelog:
            timelog.write('\n\n\nNew Model:')
            cnn = CNN(params, key_array)
            loss = cnn.cross_entropy
            loss += tf.mul(tf.constant(params['REG_STRENGTH']), cnn.reg_loss)
            train_step = cnn.optimizer.minimize(loss)
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                  intra_op_parallelism_threads=1, use_per_session_threads=True))
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(tf.all_variables())
            path = saver.save(sess, model_dir + 'temp_cnn_eval_epoch%i' %0)
            # reader = tf.train.NewCheckpointReader(path)
            # print(reader.debug_string().decode("utf-8"))
            best_dev_accuracy = cnn_eval.float_entropy(path, val_X, val_Y, key_array, params)
            timelog.write( '\ndebug acc %g' %best_dev_accuracy)
            timelog.write('\n%g'%time.clock())
            for epoch in range(params['EPOCHS']):
                batches_x, batches_y = scramble_batches(params, train_X, train_Y)
                for j in range(len(batches_x)):
                    feed_dict = {cnn.input_x: batches_x[j], cnn.input_y: batches_y[j],
                                 cnn.dropout: params['TRAIN_DROPOUT'], cnn.word_embeddings_new: np.zeros([0, key_array.shape[1]])}
                    train_step.run(feed_dict=feed_dict, session=sess)
                    #apply l2 clipping to weights and biases
                    if params['REGULARIZER'] == 'l2_clip':
                        if j == (len(batches_x) - 2):
                            print 'debug clip_vars'
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
                        else:
                            cnn.clip_vars(params)
                timelog.write('\n\nepoch %i initial time %g' %(epoch, time.clock()))
                timelog.write('\nCPU usage: %g'
                            %(resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                            resource.getrusage(resource.RUSAGE_SELF).ru_stime))
                timelog.write('\nmemory usage: %g' %(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                path = saver.save(sess, model_dir + 'temp_cnn_eval_epoch%i' %(epoch))
                dev_accuracy = cnn_eval.float_entropy(path, val_X, val_Y, key_array, params)
                timelog.write('\ndev accuracy: %g'%dev_accuracy)
                if dev_accuracy > best_dev_accuracy:
                    timelog.write('\nnew best model, epoch %i'%epoch)
                    #remove old best epoch save if exists
                    try:
                        remove_chkpt_files(path_final)
                    except (UnboundLocalError, OSError):
                        pass
                    path_final = saver.save(sess, model_dir + params['MODEL_NUM'] + '/cnn_final', global_step=epoch)
                    best_dev_accuracy = dev_accuracy
                    word_embeddings = cnn.word_embeddings.eval(session=sess)
                # elif dev_accuracy < best_dev_accuracy - .02:
                #     #remove any old chkpt files
                #     for past_epoch in range(epoch + 1):
                #         remove_chkpt_files(model_dir + 'temp_cnn_eval_epoch%i' %(past_epoch))
                #     #early stop if accuracy drops significantly
                #     try:
                #         return path_final, word_embeddings
                #     except UnboundLocalError: #path_final does not exist because initial dev accuracy highest
                #         path_final = saver.save(sess, model_dir + params['MODEL_NUM'] + '/cnn_final', global_step=epoch)
                #         return path_final, cnn.word_embeddings.eval(session=sess)
            #remove any old chkpt files
            for past_epoch in range(epoch + 1):
                try:
                    remove_chkpt_files(model_dir + 'temp_cnn_eval_epoch%i' %(past_epoch))
                except (UnboundLocalError, OSError):
                    pass
            try:
                return path_final, word_embeddings
            except UnboundLocalError: #path_final does not exist because initial dev accuracy highest
                path_final = saver.save(sess, model_dir + params['MODEL_NUM'] + '/cnn_final', global_step=epoch)
                return path_final, cnn.word_embeddings.eval(session=sess)

if __name__ == "__main__":
    main()
