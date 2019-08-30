# encoding=utf-8
import tensorflow as tf
import gdu as g
import tf_common as tc

scope = tf.variable_scope

class Transfer(object):

    def __init__(self, param, mode='train'):
        g.warn('init transfer model ... ')
        self.lr = tf.constant(param['lr'])
        self.l2_rate = param['l2_rate']
        self.emb_size = param['emb_size']
        self.hidden_size = param['hidden_size']
        self.source_vs = param['source_vocab_size']
        self.target_vs = param['target_vocab_size']
        self.encoder_layers = param['encoder_layers']
        self.decoder_layers = param['decoder_layers']
        self.mode = mode
        self.output_data = tf.placeholder(tf.int32, [None, None], name='output_data')
        self.output_teach = tf.placeholder(tf.int32, [None, None], name='output_teach')
        self.output_data_len = tf.placeholder(tf.int32, [None], name='output_data_len')
        self.input_source = tf.placeholder(tf.int32, [None, None], name='input_source')
        self.input_source_len = tf.placeholder(tf.int32, [None], name='input_source_len')
        self.batch_size = tf.size(self.input_source_len)
        self.set_global_init(tc.get_init())
        self.build()
        self.loss()
        self.optim()
        tc.show_variables()
        self.reinit = tf.initializers.variables(tc.get_scope('decoder'))
        g.normal('\nREINIT => ' + str(self.reinit))

    def set_global_init(self, init):
        tf.get_variable_scope().set_initializer(init)

    def get_noise(self, shape):
        prob = tc.runiform([], 0., 1., dtype=tf.float32)
        # if True, noise = noise_coh(0~1) * noise
        return tf.cond(prob > tf.constant(0.5),\
                       lambda: tc.runiform([], 0., 0.1, dtype=tf.float32) * tc.rnormal(shape, 0., 1., dtype=tf.float32),\
                       lambda: tc.zeros(shape, dtype=tf.float32))

    def build(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.enc_emb_table = tc.random_embeddings(self.source_vs, self.emb_size, scope_name='encoder')
        self.dec_emb_table = self.enc_emb_table
        self.enc_emb_input_source = tc.emb_lookup(self.enc_emb_table, self.input_source)
        # add noise
        # if (self.mode == 'train'):
        #     self.enc_emb_input_source += self.get_noise(tf.shape(self.enc_emb_input_source))

        with scope('encoder') as sc_enc:
            fwl, bwl = tc.bi_gru(self.hidden_size, self.encoder_layers, ac=tc.ngelu, dropout=0.05, mode=self.mode, resc=(self.encoder_layers - 1))
            sbi_out, sbi_state = tf.nn.bidirectional_dynamic_rnn(fwl, bwl, self.enc_emb_input_source, dtype=tf.float32, sequence_length=self.input_source_len)
            self.sbi_state = tc.flat_bi_states(sbi_state)

        # add noise
        if (self.mode == 'train'):
            nsshape = tf.shape(self.sbi_state[0])
            self.sbi_state = (self.sbi_state[0] + self.get_noise(nsshape),\
                              self.sbi_state[1] + self.get_noise(nsshape))

        self.dec_state = self.sbi_state
        max_encoder_length = tf.reduce_max(self.input_source_len)
        max_coh = 5.0
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * max_coh))
        decoder_initial_state = self.dec_state

        start_tokens = tf.fill([self.batch_size], 1)
        end_token = 2

        with scope('decoder') as sc_d:
            cell_p = tc.uni_gru(self.hidden_size, self.decoder_layers, ac=tc.ngelu, dropout=0.05, mode=self.mode, resc=(self.decoder_layers - 1))
            pro_p = tc.dense(self.target_vs, ac=None, use_bias=False)
            if (self.mode == 'train'):
                decoder_emb_inp = tf.nn.embedding_lookup(self.dec_emb_table, self.output_teach)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.output_data_len)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell_p, helper, decoder_initial_state)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, swap_memory=True, scope=sc_d)
                self.logits = pro_p(outputs.rnn_output)
                self.sample_id = tf.argmax(self.logits, axis=-1)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.dec_emb_table, start_tokens, end_token)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell_p, helper, decoder_initial_state, output_layer=pro_p)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations, swap_memory=True, scope=sc_d)
                self.logits = outputs.rnn_output
                self.sample_id = outputs.sample_id

        self.sample_id = tf.identity(self.sample_id, 'model_out_ids')

    def loss(self):
        self.ce_loss = tf.reduce_mean(tc.sce(labels=self.output_data, logits=self.logits))
        self.loss = self.ce_loss

    def optim(self):
        self.opt = tf.contrib.opt.AdamWOptimizer(weight_decay=self.l2_rate, learning_rate=self.lr)
        # params = tc.get_scope('embeddings') + tc.get_scope('encoder') + tc.get_scope('decoder')
        params = tc.get_scope('decoder')
        grds = tf.gradients(self.loss, params)
        grds, self.t_norm = tc.gclip(grds, maxn=2.0)
        self.update = self.opt.apply_gradients(zip(grds, params), global_step=self.global_step)

        # create saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def train(self, session, value):
        return session.run([self.update, self.loss, self.global_step, self.lr, self.t_norm], feed_dict={self.input_source:value[0], self.input_source_len:value[1], self.output_teach:value[2], self.output_data:value[3], self.output_data_len:value[4]})

    def infer(self, session, value):
        return session.run([self.sample_id], feed_dict={self.input_source:value[0], self.input_source_len:value[1]})