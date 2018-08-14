# _*_ coding: utf-8 _*_

"""
Learn `tf.variable_scope(...)` and `tf.name_scope(...)` of TensorFlow.

结论：
`tf.variable_scope(...)`可以让变量有相同的名字，包括由`tf.get_variable(...)`
和`tf.Variable(...)`创建的变量。

reference:
- https://blog.csdn.net/uestc_c2_403/article/details/72328815

Author: StrongXGP
Date:	2018/08/09
"""

import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =====================================================================
# 1. Assemble our graph												  #
# =====================================================================

with tf.variable_scope("v1"):
    a1 = tf.get_variable(name="a1", shape=[1], initializer=tf.constant_initializer(1))
    a2 = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name="a2")

with tf.variable_scope("v2"):
    a3 = tf.get_variable(name="a1", shape=[1], initializer=tf.constant_initializer(1))
    a4 = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name="a2")

# =====================================================================
# 2. Use a session to execute operations in the graph                 #
# =====================================================================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(a1.name)
    print(a2.name)
    print(a3.name)
    print(a4.name)

'''
# output:

v1/a1:0
v1/a2:0
v2/a1:0
v2/a2:0
'''
