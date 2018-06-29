# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("F:/datas/data/MNIST_data/", one_hot=True)

MAX_STEP = 30000

# show index-th image
def showImage(mnist):
    index = 1
    img = np.array(mnist.train.images[index])
    img1 = np.reshape(img, (28, 28))
    img1.astype(int)
    plt.imshow(img1)
    print(mnist.train.labels[index])



x = tf.placeholder(tf.float32, [None, 784], "x")
y = tf.placeholder(tf.float32, [None, 10], "y")

weights = tf.get_variable("weights",
                          shape = [784, 10],
                          initializer = tf.random_normal_initializer(stddev = 0.1))
biases = tf.get_variable("biases",
                         shape = [10],
                         initializer = tf.random_normal_initializer(stddev = 0.1))




out = tf.nn.bias_add(tf.matmul(x, weights), biases)
y_ = tf.nn.softmax(out)

cross_entropy = -tf.reduce_mean(y*tf.log(y_))

train_op = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

pre = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
pre = tf.cast(pre, dtype=tf.float32)
accu = tf.reduce_mean(pre)

accues = np.zeros(shape=[MAX_STEP, 1])
losses = np.zeros(shape=[MAX_STEP, 1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(MAX_STEP):
        batch_x, batch_y = mnist.train.next_batch(100)
        
        _,accur,los = sess.run([train_op, accu, cross_entropy], feed_dict={x:batch_x, y:batch_y})
        
        losses[i]=los
        accues[i]=accur
        
        if i%500 ==0:
            print(i, accur, los)
    
    print(sess.run([accu], feed_dict={x:mnist.test.images, y:mnist.test.labels}))
        
        

    
    
    






