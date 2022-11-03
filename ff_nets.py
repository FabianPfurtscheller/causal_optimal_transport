# -*- coding: utf-8 -*-
### code to output neural networks
### adapted from https://github.com/stephaneckstein/minmaxot

import tensorflow as tf

def layer(x, layernum, input_dim, output_dim, activation, outputlin=0):
    ua_w = tf.get_variable('ua_w' + str(layernum), shape=[input_dim, output_dim],
                                     initializer=tf.initializers.glorot_normal(), dtype=tf.float32)
    ua_b = tf.get_variable('ua_b' + str(layernum), shape=[output_dim],
                                     initializer=tf.initializers.glorot_normal(), dtype=tf.float32)
    if outputlin:
        z = tf.matmul(x, ua_w)
    else:
        z = tf.matmul(x, ua_w) + ua_b
    if activation.lower() == 'relu':
        return tf.nn.relu(z)
    if activation.lower() == 'tanh':
        return tf.nn.tanh(z)
    else:
        return z


def ff_net(x, name, n_layers, hidden_dim, activation, input_dim, output_dim, outputlin=0):
    with tf.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        if n_layers == 1:
            return layer(x, 0, input_dim, output_dim, activation='')
        else:
            a = layer(x, 0, input_dim, hidden_dim, activation=activation)
            for i in range(1, n_layers - 1):
                a = layer(a, i, hidden_dim, hidden_dim, activation=activation)
            a = layer(a, n_layers - 1, hidden_dim, output_dim, activation='identity', outputlin=outputlin)
            return a