import tensorflow as tf

tf.app.flags.DEFINE_integer('batch_size', 50, 'Training batch size')
tf.app.flags.DEFINE_integer('emb_size', 300, 'Size of word embeddings')
tf.app.flags.DEFINE_integer('num_kernel', 100, 'Number of filters for each window size')
tf.app.flags.DEFINE_integer('min_window', 3, 'Minimum size of filter window')
tf.app.flags.DEFINE_integer('max_window', 5, 'Maximum size of filter window')
tf.app.flags.DEFINE_integer('vocab_size', 15000, 'Vocabulary size')
tf.app.flags.DEFINE_integer('num_class', 2, 'Number of class to consider')
tf.app.flags.DEFINE_integer('sent_len', 56, 'Input sentence length. This is after the padding is performed.')
tf.app.flags.DEFINE_float('l2_reg', 0, 'l2 regularization weight')

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, initializer, wd):
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None and wd != 0.:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    else:
        weight_decay = tf.constant(0.0, dtype=tf.float32)
    return var, weight_decay

class Model(object):

    def __init__(self, config, is_train=True):
        self.is_train = is_train
        self.emb_size = config.emb_size
        self.batch_size = config.batch_size
        self.num_kernel = config.num_kernel
        self.min_window = config.min_window
        self.max_window = config.max_window
        self.vocab_size = config.vocab_size
        self.num_class = config.num_class
        self.sent_len = config.sent_len
        self.l2_reg = config.l2_reg
        if is_train:
            self.optimizer = config.optimizer
            self.dropout = config.dropout
        self.build_graph()

    def build_graph(self):
        """ Build the computation graph. """
        self._inputs = tf.placeholder(dtype=tf.int64, shape=[self.batch_size, self.sent_len], name='input_x')
        self._labels = tf.placeholder(dtype=tf.int64, shape=[self.batch_size], name='input_y')
        losses = []

        # lookup layer
        with tf.variable_scope('lookup') as scope:
            self._W_emb = _variable_on_cpu(name='embedding', shape=[self.vocab_size, self.emb_size], 
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
            # sent_batch is of shape: (batch_size, sent_len, emb_size, 1), in order to use conv2d
            sent_batch = tf.nn.embedding_lookup(params=self._W_emb, ids=self._inputs)
            sent_batch = tf.expand_dims(sent_batch, -1)

        # conv + pooling layer
        with tf.variable_scope('conv') as scope:
            pool_tensors = []
            for k_size in range(self.min_window, self.max_window+1):
                kernel, wd = _variable_with_weight_decay(name='kernel_'+str(k_size),
                    shape=[k_size, self.emb_size, 1, self.num_kernel], initializer=tf.truncated_normal_initializer(stddev=0.01), wd=self.l2_reg)
                losses.append(wd)
                conv = tf.nn.conv2d(input=sent_batch, filter=kernel, strides=[1,1,1,1], padding='VALID')
                biases = _variable_on_cpu('biases_'+str(k_size), [self.num_kernel], tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(conv, biases)
                relu = tf.nn.relu(bias, name=scope.name)
                # shape of relu: [batch_size, conv_len, 1, num_kernel]
                conv_len = relu.get_shape()[1]
                pool = tf.nn.max_pool(relu, ksize=[1,conv_len,1,1], strides=[1,1,1,1], padding='VALID')
                # shape of pool: [batch_size, 1, 1, num_kernel]
                pool = tf.squeeze(pool,squeeze_dims=[1,2]) # size: [batch_size, num_kernel]
                pool_tensors.append(pool)
            pool_layer = tf.concat(values=pool_tensors, axis=1, name='pool')

        # drop out layer
        if self.is_train and self.dropout > 0:
            pool_dropout = tf.nn.dropout(pool_layer, 1 - self.dropout)
        else:
            pool_dropout = pool_layer

        # fully-connected layer
        pool_size = (self.max_window - self.min_window + 1) * self.num_kernel
        with tf.variable_scope('fc') as scope:
            W, wd = _variable_with_weight_decay('W', shape=[pool_size, self.num_class],
                initializer=tf.truncated_normal_initializer(stddev=0.05), wd=self.l2_reg)
            losses.append(wd)
            biases = _variable_on_cpu('biases', [self.num_class], tf.constant_initializer(0.01))
            logits = tf.nn.bias_add(tf.matmul(pool_dropout, W), biases)

        # loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        losses.append(cross_entropy_loss)
        self._total_loss = tf.add_n(losses, name='total_loss')
        # self._total_loss = cross_entropy_loss

        # correct prediction count
        correct_prediction = tf.to_int32(tf.nn.in_top_k(logits, self._labels, 1))
        self._true_count_op = tf.reduce_sum(correct_prediction)

        # train on a batch
        self._lr = tf.Variable(0.0, trainable=False)
        if self.is_train:
            if self.optimizer == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(self._lr)
            elif self.optimizer == 'adagrad':
                opt = tf.train.AdagradOptimizer(self._lr)
            elif self.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(self._lr)
            elif self.optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(self._lr)
            else:
                raise ValueError("Optimizer not supported.")
            grads = opt.compute_gradients(self._total_loss)
            self._train_op = opt.apply_gradients(grads)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
        else:
            self._train_op = tf.no_op()
            
        return

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def lr(self):
        return self._lr
    
    @property
    def train_op(self):
        return self._train_op
    
    @property
    def total_loss(self):
        return self._total_loss

    @property
    def true_count_op(self):
        return self._true_count_op
    
    @property
    def W_emb(self):
        return self._W_emb

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_embedding(self, session, pretrained):
        session.run(tf.assign(self.W_emb, pretrained))
