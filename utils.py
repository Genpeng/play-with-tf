# _*_ coding: utf-8 _*_

"""
Some utility functions.

Author: StrongXGP
Date:   2018/07/03
"""

import numpy as np
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_birth_life_data(filename):
    """Read in birth life data.

    Parameters
    ----------
    filename : string, the path of data

    Returns
    -------
    data : 2d array-like, birth life data
    n_samples: int, the number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]

    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)

    return data, n_samples


def huber_loss(labels, predictions, delta=14.0):
    """Calculate the Huber loss between the true labels and the predictions.

    Parameters
    ----------
    labels : 1d array-like, the true labels of each sample
    predictions : 1d array-like, the predictions
    delta : float, a threshold for judging outliers

    Returns
    -------
    huber_loss : float, the Huber loss
    """
    residual = tf.abs(labels - predictions)

    def f1():
        return 0.5 * tf.square(residual)

    def f2():
        return delta * residual - 0.5 * tf.square(delta)

    return tf.cond(residual < delta, f1, f2)
