from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Input, Dense
from keras.models import Model
##輸入照片尺寸==28*28*1
##都用一層，288個神經元
import tensorflow as tf
import numpy as np
from functools import reduce
import operator as op
input = np.random.random_sample((28,28,1,1)).astype(np.float32)
input = tf.Variable(input)
weight= tf.Variable(tf.compat.v1.random_normal([3, 3, 1, 32]))
bias  = tf.Variable(tf.compat.v1.random_normal([32]))
conv  = tf.nn.conv2d(input, weight, strides=[1,1,1,1], padding="SAME")
print ("Total Conv parameter count: ", reduce(op.mul, weight.shape)+bias.shape[0])

input = tf.reshape(input, [1, reduce(op.mul, input.shape)])
output = tf.compat.v1.layers.dense(inputs=input, units=3*3*32, name="FC")
print ("Total FC parameter count:", input.shape[1]*(output._num_elements())+output._num_elements())
