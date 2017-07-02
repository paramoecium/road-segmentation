"""Denoising convolutional autoencoder"""

import math
import tensorflow as tf
import numpy as np

class cnn_ae():
    def __init__(self,
                 w, ## dim of the inputs,
                 learning_rate=0.005,
                 skip_connection=False):

        self.w = w
        self.learning_rate = learning_rate
        self.skip_connection = skip_connection
        self._make_graph()

    def _make_graph(self):
        tf.reset_default_graph()

        self._init_placeholders()
        self._build_graph()
        self._init_optimizer()
        self._init_summary()

    def _init_placeholders(self):
        self.x = tf.placeholder(shape=(None, self.w, self.w),
                                dtype=tf.float32,
                                name='encoder_inputs',
                                )

        x_tensor = tf.reshape(self.x, [-1, self.w, self.w, 1])
        self.x_origin = x_tensor
        self.x_noise = tf.placeholder(shape=(None, self.w, self.w),
                                      dtype=tf.float32,
                                      name='targets',
                                      )

        x_noise_tensor = tf.reshape(self.x_noise, [-1, self.w, self.w, 1])
        self.x_origin_noise = x_noise_tensor

    def weight_variable(self, shape, name):
        '''Helper function to create a weight variable initialized with
        a normal distribution
        Parameters
        ----------
        shape : list
            Size of weight variable
        '''
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape, name):
        '''Helper function to create a bias variable initialized with
        a constant value.
        Parameters
        ----------
        shape : list
            Size of weight variable
        '''
        initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'SAME')

    def deconv2d(self, x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')

    def _build_graph(self):

        if self.skip_connection:
            print("Initializing skip connection CAE")
            # encoder
            W_e_conv1 = self.weight_variable([5, 5, 1, 16], "w_e_conv1")
            b_e_conv1 = self.bias_variable([16], "b_e_conv1")
            self.h_e_conv1 = tf.nn.relu(tf.add(self.conv2d(self.x_origin_noise, W_e_conv1), b_e_conv1)) # h_e_conv1: (batch_size, 12, 12, 16)

            W_e_conv2 = self.weight_variable([5, 5, 16, 32], "w_e_conv2")
            b_e_conv2 = self.bias_variable([32], "b_e_conv2")
            self.h_e_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_e_conv1, W_e_conv2), b_e_conv2)) # h_e_conv2: (batch_size, 6, 6, 32)

            W_e_conv3 = self.weight_variable([5, 5, 32, 32], "w_e_conv2")
            b_e_conv3 = self.bias_variable([32], "b_e_conv2")
            self.h_e_conv3 = tf.nn.relu(tf.add(self.conv2d(self.h_e_conv2, W_e_conv3), b_e_conv3)) # h_e_conv2: (batch_size, 6, 6, 32)

            # decoder
            W_d_conv1 = self.weight_variable([5, 5, 32, 32], "w_d_conv1")
            b_d_conv1 = self.bias_variable([32], "b_d_conv1")
            output_shape_d_conv1 = tf.stack([tf.shape(self.x_noise)[0], 6, 6, 32])
            self.h_d_conv1 = tf.nn.relu(tf.add(
                                        self.deconv2d(self.h_e_conv3, W_d_conv1, output_shape_d_conv1),
                                        self.h_e_conv2)
                                        )

            W_d_conv2 = self.weight_variable([5, 5, 16, 32], "w_d_conv2")
            b_d_conv2 = self.bias_variable([16], "b_d_conv2")
            output_shape_d_conv2 = tf.stack([tf.shape(self.x_noise)[0], int(self.w/2), int(self.w/2), 16])
            self.h_d_conv2 = tf.nn.relu(self.deconv2d(self.h_d_conv1, W_d_conv2, output_shape_d_conv2))

            W_d_conv3 = self.weight_variable([5, 5, 1, 16], "w_d_conv2")
            b_d_conv3 = self.bias_variable([1], "b_d_conv2")
            output_shape_d_conv3 = tf.stack([tf.shape(self.x_noise)[0], self.w, self.w, 1])
            self.h_d_conv3 = tf.nn.relu(tf.add(
                                               self.deconv2d(self.h_d_conv2, W_d_conv3, output_shape_d_conv3),
                                               self.x_origin_noise)
                                               )

            self.y_pred = self.h_d_conv3
            print("reconstruct layer shape : %s" % self.y_pred.get_shape())

        else:
            """Simple 2 layer model"""
            W_e_conv1 = self.weight_variable([5, 5, 1, 16], "w_e_conv1")
            b_e_conv1 = self.bias_variable([16], "b_e_conv1")
            self.h_e_conv1 = tf.nn.relu(tf.add(self.conv2d(self.x_origin_noise, W_e_conv1), b_e_conv1)) # h_e_conv1: (batch_size, 12, 12, 16)

            W_e_conv2 = self.weight_variable([3, 3, 16, 32], "w_e_conv2")
            b_e_conv2 = self.bias_variable([32], "b_e_conv2")
            self.h_e_conv2 = tf.nn.relu(tf.add(self.conv2d(self.h_e_conv1, W_e_conv2), b_e_conv2)) # h_e_conv2: (batch_size, 6, 6, 32)

            print("code layer shape : %s" % self.h_e_conv2.get_shape())

            W_d_conv1 = self.weight_variable([5, 5, 16, 32], "w_d_conv1")
            b_d_conv1 = self.bias_variable([1], "b_d_conv1")
            output_shape_d_conv1 = tf.stack([tf.shape(self.x_noise)[0], int(self.w/2), int(self.w/2), 16])
            self.h_d_conv1 = tf.nn.relu(self.deconv2d(self.h_e_conv2, W_d_conv1, output_shape_d_conv1)) # h_d_conv1: (batch_size, 12, 12, 16)

            W_d_conv2 = self.weight_variable([5, 5, 1, 16], "w_d_conv2")
            b_d_conv2 = self.bias_variable([16], "b_d_conv2")
            output_shape_d_conv2 = tf.stack([tf.shape(self.x_noise)[0], self.w, self.w, 1])
            self.h_d_conv2 = tf.nn.relu(self.deconv2d(self.h_d_conv1, W_d_conv2, output_shape_d_conv2)) # h_d_conv2: (batch_size, 24, 24, 1)

            self.y_pred = self.h_d_conv2
            print("reconstruct layer shape : %s" % self.y_pred.get_shape())

    def _init_optimizer(self):
        self.loss = tf.reduce_mean(tf.pow(self.x_origin - self.y_pred, 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _init_summary(self):
        loss = tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge([loss])

    def make_inputs(self, data_input, data_targets):
        return {
            self.x_noise: data_input,
            self.x: data_targets,
        }

    def make_inputs_predict(self, data_input):
        return {
            self.x_noise: data_input,
        }
