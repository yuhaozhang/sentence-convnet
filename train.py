from datetime import datetime
import time
import os
import sys
import tensorflow as tf
import numpy as np

import model
import text_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data/mr/', 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', './train/', 'Directory to save training checkpoint files')
tf.app.flags.DEFINE_integer('num_epoch', 50, 'Number of epochs to run')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether log device information in summary')
# I don't know why I need to set initial lr so large here, but empirically it works pretty well
tf.app.flags.DEFINE_float('init_lr', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'LR decay rate')
tf.app.flags.DEFINE_integer('tolerance_step', 500, 'Decay the lr after loss remains unchanged for this number of steps')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate. 0 is no dropout.')
tf.app.flags.DEFINE_integer('log_step', 10, 'Write log to stdout after this step')
tf.app.flags.DEFINE_integer('summary_step', 200, 'Write summary after this step')
tf.app.flags.DEFINE_integer('save_epoch', 5, 'Save model after this epoch')

def train():
    with tf.Graph().as_default():
        # build graph
        sentences = tf.placeholder(dtype=tf.int64, shape=[FLAGS.batch_size, FLAGS.sent_len], name='input_x')
        labels = tf.placeholder(dtype=tf.int64, shape=[FLAGS.batch_size], name='input_y')
        lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')

        logits, W_emb = model.inference(sentences, keep_prob)
        loss = model.loss(logits, labels)
        train_op = model.train_batch(loss, lr)

        correct_prediction = tf.to_int32(tf.nn.in_top_k(logits, labels, 1))
        true_count_op = tf.reduce_sum(correct_prediction)

        # create a saver and summary
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
        save_path = os.path.join(FLAGS.train_dir, 'model.ckpt')

        current_lr = FLAGS.init_lr
        lowest_loss_value = float("inf")
        step_loss_ascend = 0
        global_step = 0

        # loading data
        train_loader = text_input.DataLoader(os.path.join(FLAGS.data_dir, 'train.cPickle'), batch_size=FLAGS.batch_size)
        test_loader = text_input.DataLoader(os.path.join(FLAGS.data_dir, 'test.cPickle'), batch_size=FLAGS.batch_size)
        max_steps = train_loader.num_batch * FLAGS.num_epoch # this is just an estimated number

        # loading pretrained embeddings
        pretrained_embedding = np.load(os.path.join(FLAGS.data_dir, 'emb.npy'))
        assign_op = W_emb.assign(pretrained_embedding)
        sess.run(assign_op)

        def eval_once(sess, loader):
            test_loss = 0.0
            test_accuracy = 0
            for _ in xrange(loader.num_batch):
                x_batch, y_batch = loader.next_batch()
                feed = {sentences: x_batch, labels: y_batch, keep_prob: 1.}
                loss_value, true_count_value = sess.run([loss, true_count_op], feed_dict=feed)
                test_loss += loss_value
                test_accuracy += true_count_value
            test_loss /= loader.num_batch
            test_accuracy /= (1.0 * loader.num_batch * FLAGS.batch_size)
            return (test_loss, test_accuracy)

        # Note that this is a soft version of epoch.
        for epoch in xrange(FLAGS.num_epoch):
            train_loss = 0.0
            true_count_total = 0
            for _ in xrange(train_loader.num_batch):
                global_step += 1
                start_time = time.time()
                x_batch, y_batch = train_loader.next_batch()
                feed = {sentences: x_batch, labels: y_batch, lr: current_lr, keep_prob: (1.-FLAGS.dropout)}
                _, loss_value, true_count_value = sess.run([train_op, loss, true_count_op], feed_dict=feed)
                duration = time.time() - start_time
                train_loss += loss_value
                true_count_total += true_count_value

                assert not np.isnan(loss_value), "Model loss is NaN."

                if global_step % FLAGS.log_step == 0:
                    examples_per_sec = FLAGS.batch_size / duration

                    format_str = ('%s: step %d/%d (epoch %d/%d), loss = %.6f (%.1f examples/sec; %.3f sec/batch), lr: %.6f')
                    print (format_str % (datetime.now(), global_step, max_steps, epoch+1, FLAGS.num_epoch, loss_value, 
                        examples_per_sec, duration, current_lr))

                if global_step % FLAGS.summary_step == 0:
                    summary_str = sess.run(summary_op, feed_dict=feed)
                    summary_writer.add_summary(summary_str, global_step)

                # decay learning rate if necessary
                if loss_value < lowest_loss_value:
                    lowest_loss_value = loss_value
                    step_loss_ascend = 0
                else:
                    step_loss_ascend += 1
                if step_loss_ascend >= FLAGS.tolerance_step:
                    current_lr *= FLAGS.lr_decay
                    print '%s: step %d/%d (epoch %d/%d), LR decays to %.5f' % ((datetime.now(), global_step, max_steps, 
                        epoch+1, FLAGS.num_epoch, current_lr))
                    step_loss_ascend = 0

                # stop learning if learning rate is too low
                if current_lr < 1e-5: break

            # summary loss/accuracy after each epoch
            train_loss /= train_loader.num_batch
            train_accuracy = true_count_total * 1.0 / (train_loader.num_batch * FLAGS.batch_size)
            summary_writer.add_summary(_summary_for_scalar('eval/training_loss', train_loss), global_step=epoch)
            summary_writer.add_summary(_summary_for_scalar('eval/training_accuracy', train_accuracy), global_step=epoch)

            test_loss, test_accuracy = eval_once(sess, test_loader)
            summary_writer.add_summary(_summary_for_scalar('eval/test_loss', test_loss), global_step=epoch)
            summary_writer.add_summary(_summary_for_scalar('eval/test_accuracy', test_accuracy), global_step=epoch)

            print("Epoch %d: training_loss = %.6f, training_accuracy = %.3f" % (epoch+1, train_loss, train_accuracy))
            print("Epoch %d: test_loss = %.6f, test_accuracy = %.3f" % (epoch+1, test_loss, test_accuracy))

            # save after fixed epoch
            if epoch % FLAGS.save_epoch == 0:
                    saver.save(sess, save_path, global_step=epoch)
        saver.save(sess, save_path, global_step=epoch)

def _summary_for_scalar(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])

def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()