# encoding=utf-8
import tensorflow as tf
import os
import numpy as np

import gdu as g

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

# _sentinel=None, labels=None, logits=None, dim=-1, name=None
ce = tf.nn.softmax_cross_entropy_with_logits_v2
# _sentinel=None, labels=None, logits=None, name=None
se = tf.nn.sigmoid_cross_entropy_with_logits
# labels, predictions, weights=1.0, scope=None, loss_collection='losses'
mse = tf.losses.mean_squared_error

__all__ = ['tanh', 'relu', 'relu6', 'sigmoid', 'softsign', 'softplus', 'swish', 'rnormal', 'runiform',\
           'layer', 'ce', 'se', 'mse', 'l2n', 'softmax', 'lrelu', 'get_init', 'random_embeddings',\
           'gru', 'uni_gru', 'bi_gru', 'conv2d', 'sepconv2d', 'maxpool2d', 'avgpool2d', 'dense', 'dropout',\
           'moments', 'gclip', 'load_model', 'selfatt', 'emb_lookup', 'l2_loss']

__all_anno__ = ['function tanh from tensorflow without modification.',\
                'function relu from tensorflow without modification.',\
                'function relu6 from tensorflow without modification.',\
                'function sigmoid from tensorflow without modification.',\
                'function softsign from tensorflow without modification.',\
                'function softplus from tensorflow without modification.',\
                'function swish from tensorflow without modification.',\
                'create random normal distribution tensor by tensorflow.',\
                'create random uniform distribution tensor by tensorflow.',\
                'reference from tensorflow.layers.',\
                'reference from tensorfllow.nn.softmax_cross_entropy_with_logits_v2',\
                'reference from tf.nn.sigmoid_cross_entropy_with_logits',\
                'reference from tf.losses.mean_squared_error',\
                'l2 normalize by axis.',\
                'function softmax from tensorflow without modification.',\
                'function lrelu from tensorflow without modification.',\
                'get initializer by glorot uniform or normal.',\
                'create random embeddings with defualt scop name "embeddings".',\
                'create a gru cell.',\
                'create a uni direction gru cell list.',\
                'create a bi direction gru cell list.',\
                'simple ref for conv2d from tensorflow.',\
                'simple ref for seperable conv2d from tensorflow.',\
                'simple ref for max pooling by 2d from tensorflow.',\
                'simple ref for average pooling by 2d from tensorflow.',\
                'create a dense connection.',\
                'create a dropout operation.',\
                'to calc the means and variances for tensor.',\
                'clip the gradients by specific norm value.',\
                'to chech whether need to load existed ckpt file.',\
                'a simple self-attention impl.',\
                'simple ref for tf.nn.embedding_lookup.',\
                'simple ref for tf.nn.l2_loss.']

def flist():
    '''
    List all functions and annotations within this module.
    '''
    global __all__, __all_anno__
    for f in zip(__all__, __all_anno__):
        g.normal(f)

def l2n(x, axis=-1):
    return tf.nn.l2_normalize(x, axis=axis)

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

def gru(num_units, ac=tanh, dropout=0.1, mode='train', res=False):
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

def uni_gru(num_units, num_layers, ac=tanh, dropout=0.1, mode='train', resc=0):
    gru_list = []
    for i in range(num_layers):
        gru_cell = gru(num_units, ac, dropout, mode, res=(i >= (num_layers - resc)))
        gru_list.append(gru_cell)
    if (len(gru_list) == 1):
        return gru_list[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(gru_list)

def bi_gru(num_units, num_layers, ac=tanh, dropout=0.1, mode='train', resc=0):
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

def selfatt(x, att_size, dropout_rate=0.1, mode='train'):
    if (mode != 'train'):
        dropout_rate = 0.0
    g.infor('SELFATT DROPOUT:%.2f' % dropout_rate)
    xshape = tf.shape(x)
    q = dense(att_size)(x)
    k = dense(att_size)(x)
    v = dense(att_size)(x)
    s = tf.matmul(q, k, transpose_b=True)
    s = tf.multiply(s, 1.0 / tf.sqrt(tf.cast(att_size, tf.float32)))
    s = tf.nn.softmax(s, -1)
    result = dropout(rate=dropout_rate)(tf.matmul(s, v))
    return result

def l2_reg(rate=0.0005):
    params = tf.trainable_variables()
    return tf.reduce_sum([l2_loss(v) for v in params]) * rate

def gclip(gradients, maxn=3.0):
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, maxn)
    return clipped_gradients

def load_model(model, model_dir, session, name):
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    gi = tf.global_variables_initializer()
    if latest_ckpt:
        model = model.saver.restore(session, latest_ckpt)
    else:
        session.run(gi)
        g.rainbow('CREATE NEW MODEL...')

    global_step = model.global_step.eval(session=session)
    return model, global_step
