from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np

import model
import text_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data/mr/', 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_integer('max_steps', 10000, 'Max number of steps to run')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether log device information in summary')

LOG_STEP_INTERVAL = 10
SUMMARY_STEP_INTERVAL = 100
SAVE_STEP_INTERVAL = 1000

INITIAL_LR = 0.4
LR_DECAY_RATE = 0.9
MAX_STEP_LOSS_UNCHANGED = 200
TRAIN_DROPOUT_RATE = 0.5

def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        sentences = tf.placeholder(dtype=tf.int64, shape=[FLAGS.batch_size, model.SENT_LENGTH], name='input_x')
        labels = tf.placeholder(dtype=tf.int64, shape=[FLAGS.batch_size], name='input_y')
        lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')

        logits = model.inference(sentences, keep_prob)
        loss = model.loss(logits, labels)
        train_op = model.train_batch(loss, global_step, lr)

        # create a saver and summary
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        current_lr = INITIAL_LR
        lowest_loss_value = float("inf")
        step_loss_ascend = 0

        # loading data
        reader = text_input.TextReader(FLAGS.data_dir, suffix_list=['neg', 'pos'])
        reader.prepare_data(vocab_size=FLAGS.vocab_size)
        x, y = reader.get_data_and_labels()
        loader = text_input.DataLoader(x, y, batch_size=FLAGS.batch_size)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            x_batch, y_batch = loader.next_batch()
            dict_to_feed = {sentences: x_batch, labels: y_batch, lr: current_lr, keep_prob: TRAIN_DROPOUT_RATE}
            _, loss_value = sess.run([train_op, loss], feed_dict=dict_to_feed)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), "Model loss is NaN."

            if step % LOG_STEP_INTERVAL == 0:
                examples_per_sec = FLAGS.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d/%d, loss = %.2f (%.1f examples/sec; %.3f sec/batch), lr: %.5f')
                print (format_str % (datetime.now(), step, FLAGS.max_steps, loss_value, examples_per_sec, 
                    sec_per_batch, current_lr))

            if step % SUMMARY_STEP_INTERVAL == 0:
                summary_str = sess.run(summary_op, feed_dict=dict_to_feed)
                summary_writer.add_summary(summary_str, step)

            if step % SAVE_STEP_INTERVAL == 0:
                ckpt_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, ckpt_path, global_step=step)

            # decay learning rate if necessary
            if loss_value < lowest_loss_value:
                lowest_loss_value = loss_value
                step_loss_ascend = 0
            else:
                step_loss_ascend += 1
            if step_loss_ascend >= MAX_STEP_LOSS_UNCHANGED:
                current_lr *= LR_DECAY_RATE
                print '%s: step %d/%d, LR decays to %.5f' % ((datetime.now(), step, FLAGS.max_steps, current_lr))
                step_loss_ascend = 0

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()