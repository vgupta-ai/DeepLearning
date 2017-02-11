import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#print(zip(range(0, 1280, 128),range(0, 1280, 128)))

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
#print(trX)
#trX = trX.reshape(-1, 28, 28, 1)
#teX = teX.reshape(-1, 28, 28, 1)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

w4 = init_weights([128 * 4 * 4, 625])
#print(w4.get_shape().as_list())

x = tf.placeholder('float', [128, 28,28])
y = tf.placeholder('float')

x = tf.transpose(x, [1,0,2])
print(x)
x = tf.reshape(x, [-1, 28])
print(x)
x = tf.split(0, 28, x)
print(x)