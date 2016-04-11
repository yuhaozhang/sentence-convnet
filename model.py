import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 100, 'Training batch size')
tf.app.flags.DEFINE_integer('emb_size', 300, 'Size of word embeddings')
tf.app.flags.DEFINE_integer('num_kernel', 100, 'Number of filters for each window size')
tf.app.flags.DEFINE_integer('min_window', 3, 'Minimum size of filter window')
tf.app.flags.DEFINE_integer('max_window', 5, 'Maximum size of filter window')
tf.app.flags.DEFINE_integer('vocab_size', 18000, 'Vocabulary size')

NUM_CLASSES = 2
SENT_LENGTH = 59

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _activation_summary(x, name):
    tf.histogram_summary(name + '/activations', x)
    tf.scalar_summary(name + '/sparsity', tf.nn.zero_fraction(x))
    return

def inference(sentences, keep_prob):
    """ Build the inference graph. 
        sentences is of shape: (batch_size, SENT_LEN)
    """
    # lookup layer
    with tf.variable_scope('lookup') as scope:
        W = _variable_on_cpu(name='embeddings', shape=[FLAGS.vocab_size, FLAGS.emb_size], 
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        # sent_batch is of shape: (batch_size, sent_len, emb_size, 1), in order to use conv2d
        sent_batch = tf.nn.embedding_lookup(params=W, ids=sentences)
        sent_batch = tf.expand_dims(sent_batch, -1)

    # conv + pooling layer
    with tf.variable_scope('conv') as scope:
        pool_tensors = []
        for k_size in range(FLAGS.min_window, FLAGS.max_window+1):
            kernel = _variable_on_cpu(name='kernel_'+str(k_size),
                shape=[k_size, FLAGS.emb_size, 1, FLAGS.num_kernel], initializer=tf.truncated_normal_initializer(stddev=0.01))
            conv = tf.nn.conv2d(input=sent_batch, filter=kernel, strides=[1,1,1,1], padding='VALID')
            biases = _variable_on_cpu('biases'+str(k_size), [FLAGS.num_kernel], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name=scope.name)
            # shape of relu: [batch_size, conv_len, 1, num_kernel]
            conv_len = relu.get_shape()[1]
            pool = tf.nn.max_pool(relu, ksize=[1,conv_len,1,1], strides=[1,1,1,1], padding='VALID')
            # shape of pool: [batch_size, 1, 1, num_kernel]
            pool = tf.squeeze(pool,squeeze_dims=[1,2]) # size: [batch_size, num_kernel]
            pool_tensors.append(pool)
        pool_layer = tf.concat(concat_dim=1, values=pool_tensors, name='pool')
        _activation_summary(pool_layer, name=pool_layer.name)

    # drop out layer
    pool_dropout = tf.nn.dropout(pool_layer, keep_prob)

    # fully-connected layer
    pool_size = (FLAGS.max_window - FLAGS.min_window + 1) * FLAGS.num_kernel
    with tf.variable_scope('fc') as scope:
        W = _variable_on_cpu('W', shape=[pool_size, NUM_CLASSES],
            initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
        logits = tf.nn.softmax(tf.add(tf.matmul(pool_dropout, W), biases), name=scope.name)
        _activation_summary(logits, name='softmax')

    return logits

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    return cross_entropy_loss

def train_batch(loss, global_step, lr):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    # add summary for loss, variables and gradients
    tf.scalar_summary(loss.name, loss)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    return train_op
