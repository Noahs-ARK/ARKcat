import numpy as np
import tensorflow as tf
from cnn_methods import *
from cnn_class import CNN
import cnn_eval
import time, resource
import inspect_checkpoint

#random val set empty???
#cutting 1st word --aha!
def main(params, input_X, input_Y, key_array, model_dir):

    for example in input_X:
        if not example.size:
            print 'yikes'

    train_X, train_Y, val_X, val_Y = separate_train_and_val(input_X, input_Y)

    print len (train_X), len (val_X), len(train_X) + len(val_X)
    for example in train_X:
        if not example.size:
            print 'yikes--train split'
    for example in val_X:
        if not example.size:
            print 'yikes--val split'

    cnn_dir = '../output/temp/'
    with tf.Graph().as_default():
        with open(cnn_dir + 'train_log', 'a') as timelog:
            timelog.write('\n\n\nNew Model:')
            for example in train_X:
                if np.count_nonzero(example) == 0:
                    print 'error: zero entry'
                # else:
                    # print 'maximum', np.amax(example)
                    # print 'shape', example.shape
            cnn = CNN(params, key_array)
            loss = cnn.cross_entropy
            loss += tf.mul(tf.constant(params['REG_STRENGTH']), cnn.reg_loss)
            #problem: thinks loss is None
            print loss
            train_step = cnn.optimizer.minimize(loss)
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                  intra_op_parallelism_threads=1, use_per_session_threads=True))
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(tf.all_variables())
            path = saver.save(sess, cnn_dir + 'cnn_eval_epoch%i' %0)
            # reader = tf.train.NewCheckpointReader(path)
            # print(reader.debug_string().decode("utf-8"))
            best_dev_accuracy = cnn_eval.float_entropy(path, val_X, val_Y, key_array, params)
            timelog.write( '\ndebug acc %g' %best_dev_accuracy)
            timelog.write('\n%g'%time.clock())
            for epoch in range(params['EPOCHS']):
                batches_x, batches_y = scramble_batches(params, train_X, train_Y)
                for j in range(len(batches_x)):
                    feed_dict = {cnn.input_x: batches_x[j], cnn.input_y: batches_y[j],
                                 cnn.dropout: params['TRAIN_DROPOUT']}
                    train_step.run(feed_dict=feed_dict, session=sess)
                    #apply l2 clipping to weights and biases
                    if params['REGULARIZER'] == 'l2_clip':
                        if j == (len(batches_x) - 2):
                            print 'debug clip_vars'
                            check_weights = cnn.weights[0].eval(session=sess)
                            check_biases = cnn.biases[0].eval(session=sess)
                            check_Wfc = cnn.W_fc.eval(session=sess)
                            check_bfc = cnn.b_fc.eval(session=sess)
                            cnn.clip_vars(params)
                            weights_2 = cnn.weights[0].eval(session=sess)
                            biases_2 = cnn.biases[0].eval(session=sess)
                            Wfc_2 = cnn.W_fc.eval(session=sess)
                            bfc_2 = cnn.b_fc.eval(session=sess)
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
                            print tf.nn.l2_loss(cnn.weights[0]).eval(session=sess)
                            print tf.nn.l2_loss(cnn.biases[0]).eval(session=sess)
                            print tf.nn.l2_loss(cnn.W_fc).eval(session=sess)
                            print tf.nn.l2_loss(cnn.b_fc).eval(session=sess)
                        else:
                            cnn.clip_vars(params)
                timelog.write('\n\nepoch %i initial time %g' %(epoch, time.clock()))
                timelog.write('\nCPU usage: %g'
                            %(resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                            resource.getrusage(resource.RUSAGE_SELF).ru_stime))
                timelog.write('\nmemory usage: %g' %(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                checkpoint = saver.save(sess, cnn_dir + 'cnn_eval_epoch%i' %epoch)
                dev_accuracy = cnn_eval.float_entropy(path, val_X, val_Y, key_array, params)
                timelog.write('\ndev accuracy: %g'%dev_accuracy)
                if dev_accuracy > best_dev_accuracy:
                    path = saver.save(sess, model_dir + '/cnn', global_step=epoch)
                    best_dev_accuracy = dev_accuracy
                    if dev_accuracy < best_dev_accuracy - .02:
                        #early stop if accuracy drops significantly
                        return path
            return path

if __name__ == "__main__":
    main()
