import tensorflow as tf
import numpy as np

class Model(object):
    weight = {
        'conv1': tf.Variable(tf.random_normal([2, 2, 1, 20]), name='conv1'),
        'conv2': tf.Variable(tf.random_normal([2, 2, 20, 40]), name='conv2'),
        'fc': tf.Variable(tf.random_normal([40, 2]), name='fc_weight'),
        'bias': tf.Variable(tf.constant(0.5, shape=[1, 2]), name='fc_bias')
    }
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 8, 8, 1], name="x")
        self.y = tf.placeholder(tf.float32, [None, 2], name="y")

        conv1 = tf.nn.conv2d(self.x, self.weight['conv1'], [1, 2, 2, 1], padding='VALID', name='conv1')
        relu1 = tf.nn.relu(conv1, name='relu1')
        pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool1')
        drop = tf.nn.dropout(pool1, 0.9, name='dropout')
        conv2 = tf.nn.conv2d(drop, self.weight['conv2'], [1, 2, 2, 1], padding='VALID', name='conv2')
        relu2 = tf.nn.relu(conv2, name='relu2')

        flat = tf.reshape(relu2, [-1, 40])
        self.output = tf.add(tf.matmul(flat, self.weight['fc']), self.weight['bias'])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.y))
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.loss)