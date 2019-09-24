# encoding=utf-8
import tensorflow as tf
import os
import numpy as np
import math

import gdu as g

INT32 = tf.int32
INT = INT32
FLOAT32 = tf.float32
FLOAT = FLOAT32
BOOL = tf.bool

tanh = tf.nn.tanh
relu = tf.nn.relu
relu6 = tf.nn.relu6
sigmoid = tf.nn.sigmoid
softsign = tf.nn.softsign
softplus = tf.nn.softplus
swish = tf.nn.swish
rnormal = tf.random_normal
runiform = tf.random_uniform
layer = tf.layers
l2_loss = tf.nn.l2_loss
ones = tf.ones
zeros = tf.zeros
concat = tf.concat

# _sentinel=None, labels=None, logits=None, name=None
sce = tf.nn.sparse_softmax_cross_entropy_with_logits
# _sentinel=None, labels=None, logits=None, dim=-1, name=None
ce = tf.nn.softmax_cross_entropy_with_logits_v2
# _sentinel=None, labels=None, logits=None, name=None
se = tf.nn.sigmoid_cross_entropy_with_logits
# labels, predictions, weights=1.0, scope=None, loss_collection='losses'
mse = tf.losses.mean_squared_error

__anno__ = \
'''
ANNOTATION:
name => scope name.
resc/res => residual num.
ac => activation function.
x => input tensor.
mode => train or infer mode.
'''

__all__ = [('BOOL', 'type of tensorflow.bool'),\
           ('FLOAT', 'type of tensorflow.float32'),\
           ('INT', 'type of tensorflow.int32'),\
           ('avgpool2d', 'average pool for 2d tensor, params => (pool_size, strides, padding, format, name)'),\
           ('bi_gru', 'create a bi-direction GRU cell list, params => (num_units, num_layers, ac, dropout, mode, resc)'),\
           ('ce', 'a ref of tensorflow.nn.softmax_cross_entropy_with_logits_v2, params => (_sentinel, labels, logits, dim, name)'),\
           ('conv2d', 'conv for 2d tensor, params => (filters, ksize, strides, padding, data_format, dilation_rate, ac, use_bias, name)'),\
           ('dense', 'dense layer from tensorflow, params => (units, ac, use_bias, name), {LAYER}'),\
           ('dropout', 'dropout layer from tensorflow, params => (rate, name), {LAYER}'),\
           ('dselfatt', 'self attention for dynamic length of seq2seq, params => (x, att_size, dropout_rate, res, mode)'),\
           ('emb_lookup', 'lookup the responsive emb tensor, params => (emb, source)'),\
           ('flat_bi_states', 'flat the bi-direction rnn states, params => (states, return_tuple, concat_fb)'),\
           ('gclip', 'clip the gradients, params => (gradients, maxn=2.0)'),\
           ('gelu', 'activation function from BERT, params => (x)'),\
           ('get_init', 'get tensorflow initializer, params => (type)'),\
           ('get_scope', 'get tensorflow variables by specific scope, params => (scope)'),\
           ('gru', 'GRU cell from tensorflow, params => (num_units, ac, dropout, mode, res)'),\
           ('l2_loss', 'l2 loss from tensorflow.nn.l2_loss, params => (x)'),\
           ('l2_reg', 'get l2 regularization by scope, params => (rate, scope)'),\
           ('l2n', 'l2 normalize, params => (x, axis)'),\
           ('layer', 'a ref from tensorflow.layer'),\
           ('load_model', 'load the existed model or return original model class, params => (model, model_dir, session)'),\
           ('lrelu', 'leaky relu activation function, params => (x, leak)'),\
           ('math', 'math module of python'),\
           ('maxpool2d', 'max pool for tensor, params => (pool_size, strides, padding, format, name)'),\
           ('moments', 'a ref from tensorflow.nn.moments, params => (x, axes, keep_dims)'),\
           ('mse', 'a ref from tf.losses.mean_squared_error, params => (labels, predictions, weights, scope, loss_collection)'),\
           ('ngelu', 'gelu activation function with normed, params => (x)'),\
           ('norm', 'a ref from tf.contrib.layers.layer_norm, params => (x, axis)'),\
           ('np', 'numpy module'),\
           ('ones', 'a ref from tensorflow.ones'),\
           ('random_embeddings', 'get an random embeddings table, params => (vocab_size, embedding_size, scope_name, dtype)'),\
           ('relu', 'a ref from tensorflow.nn.relu'),\
           ('relu6', 'a ref from tensorflow.nn.relu6'),\
           ('rnormal', 'a ref from tensorflow.random_normal, params => (shape, mean, var)'),\
           ('runiform', 'a ref from tensorflow.random_uniform, params => (shape, mean, var)'),\
           ('sce', 'a ref from tf.nn.sparse_softmax_cross_entropy_with_logits, params => (_sentinel, labels, logits, name)'),\
           ('se', 'a ref from tf.nn.sigmoid_cross_entropy_with_logits, params => (_sentinel, labels, logits, name)'),\
           ('sepconv2d', 'seperable conv for 2d tensor, params => (filters, ksize, strides, padding, data_format, dilation_rate, multiplier, ac, use_bias, name)'),\
           ('sigmoid', 'a ref from tensorflow.nn.sigmoid'),\
           ('softmax', 'a ref from tensorflow.nn.softmax'),\
           ('softplus', 'a ref from tensorflow.nn.softplus'),\
           ('softsign', 'a ref from tensorflow.nn.softsign'),\
           ('sself_att', 'self attention for static seq length which liked BERT(the imp of dropout has some difference with origianl paper).'),\
           ('swish', 'a ref from tensorflow.nn.swish'),\
           ('tanh', 'a ref from tensorflow.nn.tanh'),\
           ('tf', 'a ref from tensorflow'),\
           ('transformer_block', 'a implemented Transformer Block with static self attention, params => (x, layers, num_heads, num_per_heads, ac, dropout_rate, mode)'),\
           ('uni_gru', 'create an uni-direction GRU cell list, params => (num_units, num_layers, ac, dropout, mode, resc)'),\
           ('zeros', 'a ref from tensorflow.zeros'),\
           ('create_attention_mechanism', 'create attention mechnism class for luogn and bahdanau...'),\
           ('dbigru_encoder', 'create a standard seq2seq GRU encoder.'),\
           ('dgru_decoder', 'create a standard seq2seq GRU decoder with or without attention mechnism.')]

def flist():
    '''
    List all functions and annotations within this module.
    '''
    global __anno__, __all__
    g.infor(__anno__)
    for k, v in __all__:
        g.normal(k)
        g.normal('ANO => ' + v + '\n')

def l2n(x, axis=-1):
    return tf.nn.l2_normalize(x, axis=axis)

def norm(x, axis=-1):
    return tf.contrib.layers.layer_norm(inputs=x, begin_norm_axis=-1, begin_params_axis=-1)

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def ngelu(x):
    x = norm(gelu(x))
    return x

def softmax(x, axis=-1):
    return tf.nn.softmax(x, axis=axis)

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def get_init(type='uniform'):
    if (type == 'uniform'):
        return tf.contrib.keras.initializers.glorot_uniform()
    else:
        return tf.contrib.keras.initializers.glorot_normal()

def random_embeddings(vocab_size=None, embedding_size=None, scope_name='default_emb', dtype=tf.float32):
    with tf.variable_scope('embeddings') as scope:
        with tf.variable_scope(scope_name):
            v_scope_name = 'embedding_' + scope_name
            embedding_coder = tf.get_variable(v_scope_name, [vocab_size, embedding_size], dtype)
    return embedding_coder

def gru(num_units, ac=tanh, dropout=0.0, mode='train', res=False):
    if (mode != 'train'):
        dropout = 0.0
    gru = None
    gru = tf.contrib.rnn.GRUCell(num_units, activation=ac)
    if (dropout > 0.0 and dropout < 1.0):
        gru = tf.contrib.rnn.DropoutWrapper(cell=gru, input_keep_prob=(1.0 - dropout))
    if (res):
        gru = tf.contrib.rnn.ResidualWrapper(gru)
    g.infor('|  CREATE GRU UNITS:%d DROPOUT:%.2f RESIDUAL:%r  |' % (num_units, dropout, res))
    g.infor('GRU DROPOUT:%.2f' % dropout)
    return gru

def uni_gru(num_units, num_layers, ac=tanh, dropout=0.0, mode='train', resc=0):
    gru_list = []
    for i in range(num_layers):
        gru_cell = gru(num_units, ac, dropout, mode, res=(i >= (num_layers - resc)))
        gru_list.append(gru_cell)
    if (len(gru_list) == 1):
        return gru_list[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(gru_list)

def bi_gru(num_units, num_layers, ac=tanh, dropout=0.0, mode='train', resc=0):
    fw_gru_list = uni_gru(num_units, num_layers, ac, dropout, mode, resc)
    bw_gru_list = uni_gru(num_units, num_layers, ac, dropout, mode, resc)
    return fw_gru_list, bw_gru_list

def conv2d(filters, ksize=(3, 3), strides=(1, 1), padding='same',data_format='channels_last', dilation_rate=(1, 1), ac=lrelu, use_bias=True, name=None):
    return layer.Conv2D(filters=filters, kernel_size=ksize, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=ac, use_bias=use_bias, name=name)

def sepconv2d(filters, ksize=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), multiplier=1, ac=lrelu, use_bias=True, name=None):
    return layer.SeparableConv2D(filters=filters, kernel_size=ksize, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, depth_multiplier=multiplier, activation=ac, use_bias=use_bias, name=name)

def maxpool2d(pool_size=(2, 2), strides=(1, 1), padding='same', format='channels_last', name=None):
    return layer.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=format, name=name)

def avgpool2d(pool_size=(2, 2), strides=(1, 1), padding='same', format='channels_last', name=None):
    return layer.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=format, name=name)

def dense(units, ac=lrelu, use_bias=True, name=None):
    return layer.Dense(units=units, activation=ac, use_bias=use_bias, name=name)

def dropout(rate=0.1, name=None):
    return layer.Dropout(rate=rate, name=name)

def moments(x, axes=-1, keep_dims=False):
    return tf.nn.moments(x=x, axes=axes, keep_dims=keep_dims)

def emb_lookup(emb, source):
    return tf.nn.embedding_lookup(emb, source)

def dselfatt(x, att_size, dropout_rate=0.0, ac=None, res=True, mode='train'):
    if (mode != 'train'):
        dropout_rate = 0.0
    g.infor('S-SELFATT DROPOUT:%.2f' % dropout_rate)
    xshape = tf.shape(x)
    q = dense(att_size, ac=ac)(x)
    k = dense(att_size, ac=ac)(x)
    v = dense(att_size, ac=ac)(x)
    s = tf.matmul(q, k, transpose_b=True)
    s = tf.multiply(s, 1.0 / tf.sqrt(tf.cast(att_size, tf.float32)))
    s = tf.nn.softmax(s, -1)
    result = dropout(rate=dropout_rate)(tf.matmul(s, v))
    if (res):
        return result + x
    else:
        return result

def sself_att(x, num_heads, num_per_heads, ac=None, dropout_rate=0.0, mode='train'):
    '''
        x-shape: batch_size, seq_length, hidden_size/emb_size;
    '''
    if (mode != 'train'):
        dropout_rate = 0.0
    g.infor('D-SELFATT DROPOUT:%.2f' % dropout_rate)
    xshape = tf.shape(x)
    x = tf.reshape(x, [-1, xshape[-1]])
    que = dense(num_heads * num_per_heads, ac=ac)(x)
    key = dense(num_heads * num_per_heads, ac=ac)(x)
    val = dense(num_heads * num_per_heads, ac=ac)(x)
    que = tf.reshape(que, [xshape[0], num_heads, xshape[1], num_per_heads])
    key = tf.reshape(key, [xshape[0], num_heads, xshape[1], num_per_heads])
    score = tf.multiply(tf.matmul(que, key, transpose_b=True), 1.0 / math.sqrt(float(num_per_heads)))
    score = softmax(score)
    val = tf.reshape(val, [xshape[0], xshape[1], num_heads, num_per_heads])
    val = tf.transpose(val, [0, 2, 1, 3])
    context = tf.transpose(tf.matmul(score, val), [0, 2, 1, 3])
    context = tf.reshape(context, [xshape[0], xshape[1], -1])
    if (dropout_rate > 0.0):
        context = dropout(dropout_rate)(context)
    return context

def transformer_block(x, layers, num_heads, num_per_heads, ac=gelu, dropout_rate=0.0, mode='train'):
    '''
        Transformer Block, x must with a explicit seq_length and hidden_size/emb_size;
        x.shape = [batch_size, static_seq_length, hidden_size/emb_size];
    '''
    if (mode != 'train'):
        dropout_rate = 0.0
    g.infor('TRANSFORMER BLOCK DROPOUT:%.2f' % dropout_rate)
    all_layers_outputs = []
    xshape = x.shape
    layer_input = x
    for layer_idx in range(layers):
        with tf.variable_scope('layer_%d' % layer_idx):
            with tf.variable_scope("attention"):
                with tf.variable_scope("self"):
                    cxt = sself_att(layer_input, num_heads, num_per_heads, None, dropout_rate, mode)
                with tf.variable_scope("output"):
                    # linear projection
                    cxt = dense(xshape[-1].value, ac, use_bias=False)(cxt)
                    cxt = dropout(dropout_rate)(cxt)
                    cxt = norm(cxt + layer_input)
            with tf.variable_scope('intermediate'):
                itm = dense(xshape[-1].value * 4, ac)(cxt)
            with tf.variable_scope("output"):
                fout = dense(xshape[-1].value, None)(itm)
                fout = dropout(dropout_rate)(fout)
                fout = norm(fout + cxt)
                layer_input = fout
                all_layers_outputs.append(fout)
    return all_layers_outputs

def create_attention_mechanism(attention_mode='bahdanau', num_units=None, memory=None, source_sequence_length=None):
    attention_mechanism = None
    if (attention_mode == 'luong'):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif (attention_mode == 'sluong'):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length, scale=True)
    elif (attention_mode == 'bahdanau'):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif (attention_mode == 'nbahdanau'):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length, normalize=True)
    else:
        raise ValueError(g.error('INVALID ATTENTION MODE', time_tag=True, only_get=True))
    return attention_mechanism

def dbigru_encoder(hidden_size, layers, emb_input_source, input_source_len, ac=tanh,\
                   dropout_rate=0.0, mode='train', resc=0, concat_out=True, scope='BiGRU_Encoder'):
    with tf.variable_scope(scope) as sc:
        fwl, bwl = bi_gru(hidden_size, layers, ac=ac, dropout=dropout_rate, mode=mode, resc=resc)
        bi_out, bi_state = tf.nn.bidirectional_dynamic_rnn(fwl, bwl, emb_input_source, dtype=tf.float32, sequence_length=input_source_len)
    if (concat_out):
        bi_out = tf.concat(bi_out, axis=-1)
    return bi_out, bi_state

def dgru_decoder(hidden_size, layers, encoder_outputs, encoder_states, force_teach, emb_table,\
                 label, label_length, input_len, target_vocab_size, ac=tanh, dropout_rate=0.0, att_type='',\
                 mode='train', resc=0, use_states=False, max_ite=0, max_ite_rate=5.0,\
                 sos_id=1, eos_id=2, scope='UniGRU_decoder'):
    '''
    Create a standard seq2seq decoder by force teaching.
    '''
    with tf.variable_scope(scope) as sc:
        if (max_ite == 0):
            max_ite = tf.to_int32(tf.round(tf.to_float(tf.reduce_max(input_len)) * max_ite_rate))
        decoder_cell = uni_gru(hidden_size, layers, ac, dropout_rate, mode, resc)
        batch_size = tf.size(label_length)

        logits, sample_id, final_context = None, None, None
        project = dense(target_vocab_size, ac=None, use_bias=False)

        if (att_type != ''):
            attm = create_attention_mechanism(attention_mode=att_type, num_units=hidden_size, memory=encoder_outputs, source_sequence_length=input_len)
            alignment_history = True if mode != 'train' else False
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attm, attention_layer_size=hidden_size, alignment_history=alignment_history, name='attention')

        decoder_init_states = None
        if (use_states):
            decoder_init_states = encoder_states
        if (att_type != ''):
            decoder_init_states = decoder_cell.zero_state(batch_size, FLOAT)

        print 'decoder_init_states', decoder_init_states

        if (mode == 'train'):
            helper = tf.contrib.seq2seq.TrainingHelper(emb_lookup(emb_table, force_teach), label_length)
            fdecoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_init_states)
            outputs, final_context, _ = tf.contrib.seq2seq.dynamic_decode(fdecoder, swap_memory=True, scope=sc)
            logits = project(outputs.rnn_output)
            sample_id = tf.argmax(logits, axis=-1)
        else:
            start_tokens = tf.fill([batch_size], sos_id)
            end_token = eos_id
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(emb_table, start_tokens, end_token)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_init_states, output_layer=project)
            outputs, final_context, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations, swap_memory=True, scope=sc_d)
            logits = outputs.rnn_output
            sample_id = outputs.sample_id
        return logits, sample_id, final_context

def l2_reg(rate=0.0005, scope=''):
    if (scope == ''):
        params = tf.trainable_variables()
    else:
        params = get_scope(scope)
    return tf.reduce_sum([l2_loss(v) for v in params]) * rate

def gclip(gradients, maxn=2.0):
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, maxn)
    return clipped_gradients, gradient_norm

def load_model(model, model_dir, session):
    '''
    model => model class
    mode_dir => model dir string path
    session => tensorflow.Session
    '''
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    gi = tf.global_variables_initializer()
    if latest_ckpt:
        model.saver.restore(session, latest_ckpt)
    else:
        session.run(gi)
        g.infor('CREATE NEW MODEL...')

    global_step = model.global_step.eval(session=session)
    return model, global_step

def flat_bi_states(states, return_tuple=True, concat_fb=False):
    '''
    Flat the RNN states.
    e.g.: ((fw1, fw2), (bw1, bw2))
            concat_fb == False, => (fw1, bw1, fw2, bw2)
            concat_fb == True, => (fw1-bw1, fw2-bw2)
        return_tuple: [] or ()
    '''
    states_len = len(states)
    if (states_len == 2):
        rstates = []
        if (type(states[0]) != tuple):
            if (not concat_fb):
                rstates.append(states[0])
                rstates.append(states[1])
            else:
                rstates.append(tf.concat((states[0], states[1]), axis=-1))
            if (return_tuple):
                return tuple(rstates)
            else:
                return states
        else:
            # new states = [fw1, bw1, fw2, bw2 ...]
            layers = len(states[0])
            for i in range(layers):
                if (not concat_fb):
                    rstates.append(states[0][i])
                    rstates.append(states[1][i])
                else:
                    rstates.append(tf.concat((states[0][i], states[1][i]), axis=-1))
            if (return_tuple):
                return tuple(rstates)
            else:
                return rstates

def get_scope(scope=''):
    '''
    get variables from scope in {tf.GraphKeys.TRAINABLE_VARIABLES}.
    '''
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def show_variables():
    params = tf.trainable_variables()
    g.normal('\\' * 25)
    for p in params:
        g.normal(p)
    g.normal('/' * 25)