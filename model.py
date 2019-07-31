# encoding=utf-8
import tensorflow as tf
import gdu as g
import tf_common as tc

scope = tf.variable_scope

class Cls(object):

    def __init__(self, param, mode='train'):
        g.rainbow('INIT MODEL Class...')
        self.lr = param['lr']
        self.lr_drate = param['lr_drate']
        self.lr_dstep = param['lr_dstep']
        self.lr_dlimit = param['lr_dlimit']
        self.l2_rate = param['l2_rate']
        self.emb_size = param['emb_size']
        self.source_vs = param['source_vocab_size']
        self.gru_layers = param['gru_layers']
        self.cls_num = param['cls_num']
        self.mode = mode
        self.input_data = tf.placeholder(tf.int32, [None, None])
        self.input_data_len = tf.placeholder(tf.int32, [None])
        self.label = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.size(self.input_data_len)
        self.build()
        self.loss()
        self.optim()

    def set_global_init(self, init):
        tf.get_variable_scope().set_initializer(init)

    def build(self):
        # create global step
        self.global_step = tf.Variable(0, trainable=False)
        # create embedding table
        self.enc_emb_table = tc.random_embeddings(self.source_vs, self.emb_size, scope_name='encoder')
        # get specific embs
        self.enc_emb_input = tc.emb_lookup(self.enc_emb_table, self.input_data)
        self.enc_emb_input = tc.l2n(self.enc_emb_input)
        # do self attention for some iter
        with scope('wordrepr') as sc:
            satt = tc.selfatt(self.enc_emb_input, self.emb_size, mode=self.mode)
            satt = tc.l2n(satt)
            satt = tc.selfatt(satt, self.emb_size, mode=self.mode)
            satt = tc.l2n(satt)
            g.rainbow('satt')
            g.rainbow(satt)
        with scope('sentencerepr') as sc:
            # assume rnn units == emb_size
            fwl, bwl = tc.bi_gru(self.emb_size, self.gru_layers, dropout=0.1, mode=self.mode, resc=(self.gru_layers - 1))
            bi_out, bi_state = tf.nn.bidirectional_dynamic_rnn(fwl, bwl, self.enc_emb_input, dtype=tf.float32, sequence_length=self.input_data_len)
            # add the forward and backward by last layer.
            bi_state = bi_state[-1][0] + bi_state[-1][1]
            bi_state = tc.l2n(bi_state)
            g.rainbow('bi_state')
            g.rainbow(bi_state)
            bi_state = tf.expand_dims(bi_state, 1)
            bi_state = tf.tile(bi_state, [1, tf.shape(satt)[1], 1])
        with scope('combine') as sc:
            # combine the word-level representation and sentence-level representation.
            final_state = tf.concat([satt, bi_state], axis=-1)
            g.rainbow('final_state')
            g.rainbow(final_state)
        with scope('final_gru') as sc:
            ffwl, fbwl = tc.bi_gru(self.emb_size * 2, 1, dropout=0.1, mode=self.mode, resc=0)
            fbi_out, fbi_state = tf.nn.bidirectional_dynamic_rnn(ffwl, fbwl, final_state, dtype=tf.float32, sequence_length=self.input_data_len)
            fbi_state = tf.concat(fbi_state, axis=-1)
            g.rainbow('fbi_state')
            g.rainbow(fbi_state)
        with scope('projection') as sc:
            self.logits = tc.dense(self.cls_num, ac=None)(fbi_state)
            g.rainbow('logits')
            g.rainbow(self.logits)
            self.finalr = tf.argmax(self.logits, axis=-1)

    def loss(self):
        self.l2_loss = tc.l2_reg(self.l2_rate) / tf.cast(self.batch_size, tf.float32)
        ohlabel = tf.one_hot(self.label, self.cls_num)
        self.ce_loss = tf.reduce_mean(tc.ce(labels=ohlabel, logits=self.logits))
        self.loss = self.l2_loss + self.ce_loss

    def optim(self):
        self.lr = tf.maximum(tf.constant(self.lr_dlimit), tf.train.exponential_decay(self.lr, self.global_step, self.lr_dstep, self.lr_drate, staircase=True))
        self.opt = tf.train.AdamOptimizer(self.lr)
        params = tf.trainable_variables()
        grds = tf.gradients(self.loss, params)
        grds = tc.gclip(grds)
        self.update = self.opt.apply_gradients(zip(grds, params), global_step=self.global_step)
        # create saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def train(self, session, value):
        return session.run([self.update, self.ce_loss, self.l2_loss, self.global_step, self.lr], feed_dict={self.input_data: value[0], self.input_data_len: value[1], self.label: value[2]})

    def infer(self, session, value):
        return session.run([self.finalr], feed_dict={self.input_data: value[0], self.input_data_len: value[1]})