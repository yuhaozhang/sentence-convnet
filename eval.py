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

def evaluate():
    """ Build evaluation graph and run. """
    with tf.Graph().as_default():
        sentences = tf.placeholder(dtype=tf.int64, shape=[FLAGS.batch_size, FLAGS.sent_len], name='input_x')
        labels = tf.placeholder(dtype=tf.int64, shape=[FLAGS.batch_size], name='input_y')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')

        logits = model.inference(sentences, keep_prob)
        correct_prediction = tf.to_int32(tf.nn.in_top_k(logits, labels, 1))
        true_count_op = tf.reduce_sum(correct_prediction)
        loss = model.loss(logits, labels)

        saver = tf.train.Saver(tf.all_variables())

        # read test files
        x, y = text_input.load_from_dump(FLAGS.data_dir, 'test.cPickle')
        loader = text_input.DataLoader(x, y, batch_size=FLAGS.batch_size)
        num_batches = loader.batches_per_epoch()
        print 'Start evaluation, %d batches needed, with %d examples per batch.' % (num_batches, FLAGS.batch_size)

        true_count = 0
        test_loss = 0

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            for step in range(num_batches):
                x, y = loader.next_batch()
                true_count_value, loss_value = sess.run([true_count_op, loss], feed_dict={keep_prob:1.0, sentences:x, labels:y})
                true_count += true_count_value
                test_loss += loss_value

            precision = float(true_count) / loader.num_examples
            test_loss = float(test_loss) / num_batches
            print '%s: test_loss = %.6f, test_precision = %.3f' % (datetime.now(), test_loss, precision)

def main(argv=None):
    evaluate()

if __name__ == '__main__':
    tf.app.run()

