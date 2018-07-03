# _*_ coding: utf-8 _*_

"""
This is the practices of the course -- CS20: TensorFlow for Deep learning Research.

Author: StrongXGP
Date:   2018/07/03
"""

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ====================================================================== #
# Phase 1: Assemble our graph
# ====================================================================== #

# read in data from the birth_life_2010.txt file
DATA_FILE = 'data/birth_life_2010.txt'
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# create placeholders for X (birth rate) and Y (life expectancy)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# create weights and biases, initialized to 0.0
w = tf.get_variable(name="weights", initializer=tf.constant(0.0))
b = tf.get_variable(name="bias", initializer=tf.constant(0.0))

# build model to predict Y
Y_predicted = tf.add(tf.multiply(w, X), b, name="Y_predicted")

# use the square error as the loss function
loss = tf.reduce_mean(tf.square(Y - Y_predicted), name="loss")

# using gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# ====================================================================== #
# Phase 2: Use a session to execute operations in the graph
# ====================================================================== #

start_time = time.time()

# create a `FileWriter` object to write the model's graph to TensorBoard
writer = tf.summary.FileWriter("./graphs/03/linear_regression", tf.get_default_graph())

with tf.Session() as sess:
    # initialize the necessary variables
    sess.run(tf.global_variables_initializer())

    # train the model for 100 epochs
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, loss_out = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += loss_out

        print("Epoch {0}: {1}".format(i, total_loss/n_samples))

    # close the writer when you're done using it
    writer.close()

    # output the values of w and b
    w_out, b_out = sess.run([w, b])

print("Took: %f seconds" % (time.time() - start_time))
