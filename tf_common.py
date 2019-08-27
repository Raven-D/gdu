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
           ('bahdanau_att', 'create a bahdanau attention with wrapper, params => (cell, units, memory, seq_len, normed, align_history, name)'),\
           ('bi_gru', 'create a bi-direction GRU cell list, params => (num_units, num_layers, ac, dropout, mode, resc)'),\
           ('ce', 'a ref of tensorflow.nn.softmax_cross_entropy_with_logits_v2, params => (_sentinel, labels, logits, dim, name)'),\
           ('conv2d', 'conv for 2d tensor, params => (filters, ksize, strides, padding, data_format, dilation_rate, ac, use_bias, name)'),\
           ('dense', 'dense layer from tensorflow, params => (units, ac, use_bias, name), {LAYER}'),\
           ('dropout', 'dropout layer from tensorflow, params => (rate, name), {LAYER}'),\
           ('dselfatt', 'self attention for dynamic length of seq2seq, params => (x, att_size, dropout_rate, res, mode)'),\
           ('emb_lookup', 'lookup the responsive emb tensor, params => (emb, source)'),\
           ('flat', 'flat a muilti-rank(<=2) iterable item to a python list, params => (x, name, convert_to_tensor)'),\
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
           ('luong_att', 'create luong attention, params => (cell, units, memory, seq_len, scaled, align_history, name)'),\
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
           ('zeros', 'a ref from tensorflow.zeros')]

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
    return layer.Conv2D(filters=filters, ksize=ksize, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=ac, use_bias=use_bias, name=name)

def sepconv2d(filters, ksize=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), multiplier=1, ac=lrelu, use_bias=True, name=None):
    return layer.SeparableConv2D(filters=filters, kernel_size=ksize, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, depth_multiplier=multiplier, activation=ac, use_bias=use_bias, name=name)

def maxpool2d(pool_size=(2, 2), strides=(1, 1), padding='same', format='channel_last', name=None):
    return layer.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=format, name=name)

def avgpool2d(pool_size=(2, 2), strides=(1, 1), padding='same', format='channel_last', name=None):
    return layer.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=format, name=name)

def dense(units, ac=lrelu, use_bias=True, name=None):
    return layer.Dense(units=units, activation=ac, use_bias=use_bias, name=name)

def dropout(rate=0.1, name=None):
    return layer.Dropout(rate=rate, name=name)

def moments(x, axes=-1, keep_dims=False):
    return tf.nn.moments(x=x, axes=axes, keep_dims=keep_dims)

def emb_lookup(emb, source):
    return tf.nn.embedding_lookup(emb, source)

def dselfatt(x, att_size, dropout_rate=0.0, res=True, mode='train'):
    if (mode != 'train'):
        dropout_rate = 0.0
    g.infor('S-SELFATT DROPOUT:%.2f' % dropout_rate)
    xshape = tf.shape(x)
    q = dense(att_size)(x)
    k = dense(att_size)(x)
    v = dense(att_size)(x)
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

def l2_reg(rate=0.0005, scope=''):
    if (scope == ''):
        params = tf.trainable_variables()
    else:
        params = get_scope(scope)
    return tf.reduce_sum([l2_loss(v) for v in params]) * rate

def luong_att(cell, units, memory, seq_len, scaled=False, align_history=False, name='luong_attention'):
    att = tf.contrib.seq2seq.LuongAttention(units, memory, memory_sequence_length=seq_len, scale=scaled)
    return tf.contrib.seq2seq.AttentionWrapper(cell, att, attention_layer_size=units, alignment_history=align_history, name=name)

def bahdanau_att(cell, units, memory, seq_len, normed=False, align_history=False, name='bahdanau_attention'):
    att = tf.contrib.seq2seq.BahdanauAttention(units, memory, memory_sequence_length=seq_len, normalize=normed)
    return tf.contrib.seq2seq.AttentionWrapper(cell, att, attention_layer_size=units, alignment_history=align_history, name=name)

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

def flat(x, name='flat', convert_to_tensor=False):
    '''
    Flat up to 2 layers of iterable elements.
    e.g.: [(a,b), (c,d)] => [a,b,c,d].
    '''
    r = []
    tx = type(x)
    for e1 in x:
        tex = type(e1)
        if (tex == list or tex == tuple):
            for e2 in e1:
                r.append(e2)
        else:
            r.append(e1)
    if (not convert_to_tensor):
        return r
    return tf.identity(r, name)

def get_scope(scope=''):
    '''
    get variables from scope in {tf.GraphKeys.TRAINABLE_VARIABLES}.
    '''
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)