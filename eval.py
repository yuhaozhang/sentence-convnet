from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np

import model
import text_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data/mr/', 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Where to read model')
tf.app.flags.DEFINE_boolean('train_data', False, 'To evaluate on training data')

def evaluate():
    """ Build evaluation graph and run. """
    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            m = model.Model(FLAGS, is_train=False)
        saver = tf.train.Saver(tf.global_variables())

        # read test files
        if FLAGS.train_data:
            loader = text_input.DataLoader(os.path.join(FLAGS.data_dir, 'train.cPickle'), batch_size=FLAGS.batch_size)
        else:
            loader = text_input.DataLoader(os.path.join(FLAGS.data_dir, 'test.cPickle'), batch_size=FLAGS.batch_size)
        print 'Start evaluation, %d batches needed, with %d examples per batch.' % (loader.num_batch, FLAGS.batch_size)

        true_count = 0
        avg_loss = 0

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            for _ in range(loader.num_batch):
                x, y = loader.next_batch()
                true_count_value, loss_value = sess.run([m.true_count_op, m.total_loss], 
                    feed_dict={m.inputs:x, m.labels:y})
                true_count += true_count_value
                avg_loss += loss_value

            accuracy = float(true_count) / (loader.num_batch * FLAGS.batch_size)
            avg_loss = float(avg_loss) / loader.num_batch
            print '%s: test_loss = %.6f, test_accuracy = %.3f' % (datetime.now(), avg_loss, accuracy)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()

