import numpy as np
import tensorflow as tf
from cnn_methods import *
from cnn_class import CNN
import cnn_eval
import time, resource
import inspect_checkpoint

#random val set empty???
def main(params, train_X, train_Y, key_array, model_dir):
    #
    # for example in train_X:
    #     if not example.size:
    #         print 'yikes'
    #
    train_X, train_Y, val_X, val_Y = separate_train_and_val(train_X, train_Y)

    # print len (train_X), len (val_X), len(train_X) + len(val_X)
    # for example in train_X:
    #     if not example.size:
    #         print 'yikes--train split'
    # for example in train_X:
    #     if not example.size:
    #         print 'yikes--val split'
    #
    cnn_dir = '../output/temp/'
    with tf.Graph().as_default():
        with open(cnn_dir + 'train_log', 'a') as timelog:
            timelog.write('\n\n\nNew Model:')

            cnn = CNN(params, key_array)
            loss = cnn.cross_entropy
            loss += tf.mul(tf.constant(params['REG_STRENGTH']), cnn.reg_loss)
            train_step = cnn.optimizer.minimize(loss)
            sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,
                                  intra_op_parallelism_threads=1, use_per_session_threads=True))
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(tf.all_variables())
            path = saver.save(sess, cnn_dir + 'cnn_eval_%s_epoch%i' %(params['model_num'], 0))
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
                        cnn.clip_vars(params)
                timelog.write('\n\nepoch %i initial time %g' %(epoch, time.clock()))
                timelog.write('\nCPU usage: %g'
                            %(resource.getrusage(resource.RUSAGE_SELF).ru_utime +
                            resource.getrusage(resource.RUSAGE_SELF).ru_stime))
                timelog.write('\nmemory usage: %g' %(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                checkpoint = saver.save(sess, cnn_dir + 'cnn_eval_%s_epoch%i' %(params['model_num'], epoch))
                dev_accuracy = cnn_eval.float_entropy(path, val_X, val_Y, key_array, params)
                timelog.write('\ndev accuracy: %g'%dev_accuracy)
                if dev_accuracy > best_dev_accuracy:
                    path = saver.save(sess, model_dir + '/cnn_' + params['model_num'], global_step=epoch)
                    best_dev_accuracy = dev_accuracy
                    if dev_accuracy < best_dev_accuracy - .02:
                        #early stop if accuracy drops significantly
                        return path
            return path

if __name__ == "__main__":
    main()
