# coding=utf-8

import tensorflow as tf
from util.Randomer import Randomer


class GatedLayer(object):
    '''
    门限层
    '''

    def __init__(self, hidden_size, stddev=None):
        '''
        hidden_size: int,
        '''
        self.stddev = stddev
        self.hidden_size = hidden_size
        self.W = tf.Variable(
            Randomer.random_normal(
                [self.hidden_size, self.hidden_size]),
            trainable=True
        )
        self.U = tf.Variable(
            Randomer.random_normal(
                [self.hidden_size, self.hidden_size]),
            trainable=True
        )
        self.b = tf.Variable(tf.zeros([1]), trainable=True)

    def count_alpha(self, xa, xb):
        '''
        计算门限
        Args:
            xa: shape = [batch_size, time_step, hidden_size]
            xb: shape = [batch_size, time_step, hidden_size]
        Returns:
            alpah: shape=[bathc_size, time_step, hidden_size]
        '''
        batch_size = tf.shape(xa)[0]
        W_3dim = tf.reshape(
            tf.tile(self.W, [batch_size, 1]),
            [batch_size, self.hidden_size, self.hidden_size]
        )
        U_3dim = tf.reshape(
            tf.tile(self.U, [batch_size, 1]),
            [batch_size, self.hidden_size, self.hidden_size]
        )
        alpha = tf.add(
            tf.add(
                tf.matmul(xa, W_3dim),
                tf.matmul(xb, U_3dim)
            ),
            self.b
        )
        alpha = tf.sigmoid(alpha)

        return alpha

    def forward(self, xa, xb, xc=None):
        '''
        计算xa与xb的门限输出
        Args:
            xa: shape = [batch_size, time_step, hidden_size]
            xb: shape = [bathc_size, time_step, hidden_size]
        Returns:
            tensor of shape [batch_size, time_step, hidden_size]
        '''
        alpha = self.count_alpha(xa, xb)
        outputs = (1 - alpha) * xa + alpha * xb
        return outputs
