import tensorflow as tf
import math


class Model(object):
    def __init__(self, hidden_size=100, out_size=100, batch_size=100, nonhybrid=True):
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.mask = tf.placeholder(dtype=tf.float32)
        self.pre_mask = tf.placeholder(dtype=tf.float32)
        self.alias = tf.placeholder(dtype=tf.int32)
        self.pre_alias = tf.placeholder(dtype=tf.int32)
        self.item = tf.placeholder(dtype=tf.int32)
        self.tar = tf.placeholder(dtype=tf.int32)
        self.user = tf.placeholder(dtype=tf.int32)
        self.nonhybrid = nonhybrid
        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.pre = tf.placeholder(dtype=tf.int32)

        self.nasr_w1 = tf.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v2 = tf.get_variable('nasrv2', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.nasr_b2 = tf.get_variable('nasr_b2', [self.out_size], dtype=tf.float32, initializer=tf.zeros_initializer())

    def attention_level_one(self, user_embedding, pre_sessions_embedding):
        weight = tf.nn.softmax(tf.transpose(tf.matmul(pre_sessions_embedding, tf.transpose(user_embedding))))
        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.transpose(weight)), axis=0)
        return out

    def attention_level_two(self, user_embedding, long_user_embedding, current_session_embedding):

        weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
            tf.transpose(user_embedding))))

        out = tf.reduce_sum(
            tf.multiply(tf.concat([current_session_embedding, tf.expand_dims(long_user_embedding, axis=0)], 0),
                        tf.transpose(weight)), axis=0)
        return out

    def forward(self, re_embedding, user_matrix, item_matrix, train=True):
        user_embedding = tf.nn.embedding_lookup(user_matrix, self.user)
        user_embedding = tf.reshape(user_embedding, [self.batch_size, self.out_size])

        pre_embedding = tf.nn.embedding_lookup(item_matrix, self.pre)
        pre_embedding = tf.reshape(pre_embedding, [self.batch_size, -1, self.out_size])
        seq_pre = tf.stack([tf.nn.embedding_lookup(pre_embedding[i], self.pre_alias[i]) for i in range(self.batch_size)],
                         axis=0)  # batch_size*T*d
        seq = tf.matmul(tf.reshape(seq_pre, [-1, self.out_size]), self.nasr_w1)
        m = tf.nn.sigmoid(tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)
        coef1 = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(
            self.pre_mask, [-1, 1])


        rm = tf.reduce_sum(self.mask, 1)
        last_id = tf.gather_nd(self.alias, tf.stack([tf.range(self.batch_size), tf.to_int32(rm) - 1], axis=1))
        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], self.alias[i]) for i in range(self.batch_size)],
                         axis=0)  # batch_size*T*d
        # last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        # last = tf.reshape(last, [self.batch_size, 1, -1])
        m = tf.nn.sigmoid(tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b2)
        coef2 = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v2, transpose_b=True) * tf.reshape(
            self.mask, [-1, 1])



        b = self.embedding[1:]
        if not self.nonhybrid:
            ma = tf.concat([tf.reduce_sum(tf.reshape(coef1, [self.batch_size, -1, 1]) * seq_pre, 1),
                            tf.reduce_sum(tf.reshape(coef2, [self.batch_size, -1, 1]) * seq_h, 1)], -1)

            self.B = tf.get_variable('B', [2 * self.out_size, self.out_size],
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
            y1 = tf.matmul(ma, self.B)
            logits = tf.matmul(y1, b, transpose_b=True)
        else:
            ma = tf.reduce_sum(tf.reshape(coef2, [self.batch_size, -1, 1]) * seq_h, 1)
            logits = tf.matmul(ma, b, transpose_b=True)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.tar - 1, logits=logits))
        self.vars = tf.trainable_variables()
        if train:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if v.name not
                               in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.L2
            loss = loss + lossL2
        return loss, logits

    def run(self, fetches, tar, item, user, pre, adj_in, adj_out, alias, mask, pre_alias, pre_mask):
        return self.sess.run(fetches, feed_dict={self.tar: tar, self.item: item, self.adj_in: adj_in, self.user: user,
                                                 self.pre: pre, self.adj_out: adj_out, self.alias: alias,
                                                 self.mask: mask, self.pre_alias: pre_alias, self.pre_mask: pre_mask})


class GGNN(Model):
    def __init__(self, hidden_size=100, out_size=100, batch_size=300, n_node=None, n_user=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False):
        super(GGNN, self).__init__(hidden_size, out_size, batch_size, nonhybrid)
        self.embedding = tf.get_variable(shape=[n_node, hidden_size], name='embedding', dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.user_embedding = tf.get_variable(shape=[n_user, hidden_size], name='user_embedding', dtype=tf.float32,
                                              initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.adj_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None])
        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        self.W_in = tf.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(-self.stdv, self.stdv))
        with tf.variable_scope('ggnn_model', reuse=None):
            re_embedding, user_embedding, item_embedding = self.ggnn()
            self.loss_train, _ = self.forward(re_embedding, user_embedding, item_embedding)
        with tf.variable_scope('ggnn_model', reuse=True):
            re_embedding, user_embedding, item_embedding = self.ggnn()
            self.loss_test, self.score_test = self.forward(re_embedding, user_embedding, item_embedding, train=False)
        self.global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(lr, global_step=self.global_step, decay_steps=decay,
                                                        decay_rate=lr_dc, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_train, global_step=self.global_step)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def ggnn(self):
        fin_state = tf.nn.embedding_lookup(self.embedding, self.item)
        cell = tf.contrib.rnn.GRUCell(self.out_size)
        # cell = tf.nn.rnn_cell.GRUCell(self.out_size)
        with tf.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(self.adj_in, fin_state_in),
                                tf.matmul(self.adj_out, fin_state_out)], axis=-1)
                state_output, fin_state = \
                    tf.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2 * self.out_size]), axis=1),
                                      initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        return tf.reshape(fin_state, [self.batch_size, -1, self.out_size]), self.user_embedding, self.embedding
