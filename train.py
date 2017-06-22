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
tf.app.flags.DEFINE_boolean('use_pretrain', True, 'Use word2vec pretrained embeddings or not')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether log device information in summary')

tf.app.flags.DEFINE_string('optimizer', 'adagrad', 'Optimizer to use. Must be one of "sgd", adagrad", "adadelta" and "adam"')
tf.app.flags.DEFINE_float('init_lr', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'LR decay rate')
tf.app.flags.DEFINE_integer('tolerance_step', 500, 'Decay the lr after loss remains unchanged for this number of steps')
tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate. 0 is no dropout.')

tf.app.flags.DEFINE_integer('log_step', 10, 'Write log to stdout after this step')
tf.app.flags.DEFINE_integer('summary_step', 200, 'Write summary after this step')
tf.app.flags.DEFINE_integer('save_epoch', 5, 'Save model after this epoch')

def train():
    # load data
    train_loader = text_input.DataLoader(os.path.join(FLAGS.data_dir, 'train.cPickle'), batch_size=FLAGS.batch_size)
    test_loader = text_input.DataLoader(os.path.join(FLAGS.data_dir, 'test.cPickle'), batch_size=FLAGS.batch_size)
    max_steps = train_loader.num_batch * FLAGS.num_epoch # this is just an estimated number

    with tf.Graph().as_default():
        with tf.variable_scope('cnn', reuse=None):
            m = model.Model(FLAGS, is_train=True)
        with tf.variable_scope('cnn', reuse=True):
            mtest = model.Model(FLAGS, is_train=False)

        saver = tf.train.Saver(tf.global_variables())
        save_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        summary_op = tf.summary.merge_all()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        if FLAGS.use_pretrain:
            print "Use pretrained embeddings to initialize model ..."
            pretrained_embedding = np.load(os.path.join(FLAGS.data_dir, 'emb.npy'))
            m.assign_embedding(sess, pretrained_embedding)

        current_lr = FLAGS.init_lr
        lowest_loss_value = float("inf")
        step_loss_ascend = 0
        global_step = 0

        def eval_once(mtest, sess, data_loader):
            test_loss = 0.0
            test_accuracy = 0
            for _ in xrange(data_loader.num_batch):
                x_batch, y_batch = data_loader.next_batch()
                x_batch = np.array(x_batch)
                loss_value, true_count_value = sess.run([mtest.total_loss, mtest.true_count_op], 
                    feed_dict={mtest.inputs: x_batch, mtest.labels: y_batch})
                test_loss += loss_value
                test_accuracy += true_count_value
            test_loss /= data_loader.num_batch
            test_accuracy /= (1.0 * data_loader.num_batch * FLAGS.batch_size)
            data_loader.reset_pointer()
            return (test_loss, test_accuracy)

        # Note that this is a soft version of epoch.
        for epoch in xrange(FLAGS.num_epoch):
            train_loss = 0.0
            true_count_total = 0
            train_loader.reset_pointer()
            for _ in xrange(train_loader.num_batch):
                m.assign_lr(sess, current_lr)
                global_step += 1
                start_time = time.time()
                x_batch, y_batch = train_loader.next_batch()
                feed = {m.inputs: x_batch, m.labels: y_batch}
                _, loss_value, true_count_value = sess.run([m.train_op, m.total_loss, m.true_count_op], feed_dict=feed)
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
                    summary_str = sess.run(summary_op)
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

            test_loss, test_accuracy = eval_once(mtest, sess, test_loader)
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
