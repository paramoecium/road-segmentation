"""Helper funcions for model_weightedloss.py"""

import numpy as np
import tensorflow as tf

def calculate_balancing_weights(train_labels):
    """
    Calculates the class weights that are used for countering class imbalance in the loss function.
    :param train_labels: List of one-hot patch labels, with background
           being the first component and road the second.
    :return: An array with first the weight for the background
             class and second the weight for the road class.
    """

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 += 1
        else:
            c1 += 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))

    # We apply median frequency class balancing as in arXiv:1411.4734v4
    c0freq = float(c0) / (c0 + c1)
    c1freq = float(c1) / (c0 + c1)
    medianfreq = (c0freq + c1freq) / 2 # We only have two classes, so avg==median
    return np.array([medianfreq / c0freq, medianfreq / c1freq])

def weighted_softmax_crossentropy_loss(logits, labels, weights=np.array([1,1])):
    """
    Adaption of the tf.nn.softmax_cross_entropy_with_logits that supports class weighting.

    Inspired by https://blog.fineighbor.com/tensorflow-dealing-with-imbalanced-data-eb0108b10701

    :param logits: Unscaled log probabilities.
    :param labels: The groundtruth.
    :param weights: Class weights to be
    :return: The element wise cross-entropy
    """

    with tf.name_scope('loss_1'):
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon
        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), weights), reduction_indices=[1])
        return cross_entropy
