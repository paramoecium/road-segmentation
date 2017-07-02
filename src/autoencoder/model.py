""" Denoising Auto Encoder
ref: https://raw.githubusercontent.com/aymericdamien/TensorFlow-Examples/master/examples/3_NeuralNetworks/autoencoder.py
"""
import math
import tensorflow as tf
import numpy as np

class ae():
    """Autoencoder class"""
    def __init__(self,
                 n_input,
                 n_hidden_1,
                 n_hidden_2,
                 stack_1,
                 stack_2,
                 learning_rate,
                 dropout=1.0,
                 skip_arch=False):

        # Network Parameters
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_input = n_input

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.skip_arch = skip_arch

        self.stack_1 = stack_1
        self.stack_2 = stack_2

        self._make_graph()

    def _make_graph(self):
        tf.reset_default_graph()

        self._init_placeholders()
        self._init_network_params()

        self._init_encoder()
        if self.skip_arch:
            self._init_decoder_skip_arch()
        else:
            self._init_decoder()
        if self.stack_1:
            self._stack_1()
        if self.stack_2:
            self._stack_2()

        self._init_optimizer()
        self._init_summary()

    def _init_placeholders(self):
        self.x = tf.placeholder(shape=(None, self.n_input),
                                dtype=tf.float32,
                                name='encoder_inputs',
                                )

        self.y = tf.placeholder(shape=(None, self.n_input),
                                dtype=tf.float32,
                                name='targets',
                                )

    def _init_network_params(self):
        ## Uniform(-sqrt(3), sqrt(3)) has variance=1.
        sqrt3 = math.sqrt(3)
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

        self.weights = {
            'encoder_h1': tf.get_variable(name="encoder_w_h1",
                                          shape=[self.n_input, self.n_hidden_1],
                                          initializer=initializer,
                                          dtype=tf.float32),
            'encoder_h2': tf.get_variable(name="encoder_w_h2",
                                          shape=[self.n_hidden_1, self.n_hidden_2],
                                          initializer=initializer,
                                          dtype=tf.float32),
            'decoder_h1': tf.get_variable(name="decoder_w_h1",
                                          shape=[self.n_hidden_2, self.n_hidden_1],
                                          initializer=initializer,
                                          dtype=tf.float32),
            'decoder_h2': tf.get_variable(name="decoder_w_h2",
                                          shape=[self.n_hidden_1, self.n_input],
                                          initializer=initializer,
                                          dtype=tf.float32),
        }
        self.biases = {
            'encoder_b1': tf.get_variable(name="encoder_b_h1",
                                          shape=[self.n_hidden_1],
                                          initializer=initializer,
                                          dtype=tf.float32),
            'encoder_b2': tf.get_variable(name="encoder_b_h2",
                                          shape=[self.n_hidden_2],
                                          initializer=initializer,
                                          dtype=tf.float32),
            'decoder_b1': tf.get_variable(name="decoder_b_h1",
                                          shape=[self.n_hidden_1],
                                          initializer=initializer,
                                          dtype=tf.float32),
            'decoder_b2': tf.get_variable(name="decoder_b_h2",
                                          shape=[self.n_input],
                                          initializer=initializer,
                                          dtype=tf.float32),
        }

        if self.stack_1:
            self.stack_1_weights = {
                's1_encoder_h1': tf.get_variable(name="s1_encoder_w_h1",
                                              shape=[self.n_input, self.n_hidden_1],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's1_encoder_h2': tf.get_variable(name="s1_encoder_w_h2",
                                              shape=[self.n_hidden_1, self.n_hidden_2],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's1_decoder_h1': tf.get_variable(name="s1_decoder_w_h1",
                                              shape=[self.n_hidden_2, self.n_hidden_1],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's1_decoder_h2': tf.get_variable(name="s1_decoder_w_h2",
                                              shape=[self.n_hidden_1, self.n_input],
                                              initializer=initializer,
                                              dtype=tf.float32),
            }
            self.stack_1_biases = {
                's1_encoder_b1': tf.get_variable(name="s1_encoder_b_h1",
                                              shape=[self.n_hidden_1],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's1_encoder_b2': tf.get_variable(name="s1_encoder_b_h2",
                                              shape=[self.n_hidden_2],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's1_decoder_b1': tf.get_variable(name="s1_decoder_b_h1",
                                              shape=[self.n_hidden_1],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's1_decoder_b2': tf.get_variable(name="s1_decoder_b_h2",
                                              shape=[self.n_input],
                                              initializer=initializer,
                                              dtype=tf.float32),
            }

        if self.stack_2:
            self.stack_2_weights = {
                's2_encoder_h1': tf.get_variable(name="s2_encoder_w_h1",
                                              shape=[self.n_input, self.n_hidden_1],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's2_encoder_h2': tf.get_variable(name="s2_encoder_w_h2",
                                              shape=[self.n_hidden_1, self.n_hidden_2],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's2_decoder_h1': tf.get_variable(name="s2_decoder_w_h1",
                                              shape=[self.n_hidden_2, self.n_hidden_1],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's2_decoder_h2': tf.get_variable(name="s2_decoder_w_h2",
                                              shape=[self.n_hidden_1, self.n_input],
                                              initializer=initializer,
                                              dtype=tf.float32),
            }
            self.stack_2_biases = {
                's2_encoder_b1': tf.get_variable(name="s2_encoder_b_h1",
                                              shape=[self.n_hidden_1],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's2_encoder_b2': tf.get_variable(name="s2_encoder_b_h2",
                                              shape=[self.n_hidden_2],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's2_decoder_b1': tf.get_variable(name="s2_decoder_b_h1",
                                              shape=[self.n_hidden_1],
                                              initializer=initializer,
                                              dtype=tf.float32),
                's2_decoder_b2': tf.get_variable(name="s2_decoder_b_h2",
                                              shape=[self.n_input],
                                              initializer=initializer,
                                              dtype=tf.float32),
            }

    # Building the encoder
    def _init_encoder(self):
        with tf.variable_scope("encoder") as scope:
            self.layer_1 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.x, self.weights['encoder_h1']),
                                                    self.biases['encoder_b1'])), keep_prob=self.dropout)

            self.layer_2 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.layer_1, self.weights['encoder_h2']),
                                                    self.biases['encoder_b2'])), keep_prob=self.dropout)

            # self.layer_3 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.layer_2, self.weights['encoder_h3']),
            #                                         self.biases['encoder_b3'])), keep_prob=self.dropout)

            # self.layer_4 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.layer_3, self.weights['encoder_h4']),
            #                                         self.biases['encoder_b4'])), keep_prob=self.dropout)

    # Building the decoder
    def _init_decoder(self):
        with tf.variable_scope("decoder") as scope:
            self.layer_5 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.layer_2, self.weights['decoder_h1']),
                                                    self.biases['decoder_b1'])), keep_prob=self.dropout)

            self.layer_6 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.layer_5, self.weights['decoder_h2']),
                                                    self.biases['decoder_b2'])), keep_prob=self.dropout)

            # self.layer_7 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.layer_6, self.weights['decoder_h3']),
            #                                         self.biases['decoder_b3'])), keep_prob=self.dropout)

            # self.layer_8 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.layer_7, self.weights['decoder_h4']),
            #                                         self.biases['decoder_b4'])), keep_prob=self.dropout)
    def _stack_1(self):
        with tf.variable_scope("stack_1") as scope:
            stack_1_layer_1 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.layer_6, self.stack_1_weights['s1_encoder_h1']),
                                                    self.stack_1_biases['s1_encoder_b1'])), keep_prob=self.dropout)

            stack_1_layer_2 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(stack_1_layer_1, self.stack_1_weights['s1_encoder_h2']),
                                                    self.stack_1_biases['s1_encoder_b2'])), keep_prob=self.dropout)

            stack_1_layer_3 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(stack_1_layer_2, self.stack_1_weights['s1_decoder_h1']),
                                                    self.stack_1_biases['s1_decoder_b1'])), keep_prob=self.dropout)

            self.stack_1_layer_4 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(stack_1_layer_3, self.stack_1_weights['s1_decoder_h2']),
                                                    self.stack_1_biases['s1_decoder_b2'])), keep_prob=self.dropout)

    def _stack_2(self):
        with tf.variable_scope("stack_2") as scope:
            stack_2_layer_1 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.stack_1_layer_4, self.stack_2_weights['s2_encoder_h1']),
                                                    self.stack_2_biases['s2_encoder_b1'])), keep_prob=self.dropout)

            stack_2_layer_2 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(stack_2_layer_1, self.stack_2_weights['s2_encoder_h2']),
                                                    self.stack_2_biases['s2_encoder_b2'])), keep_prob=self.dropout)

            stack_2_layer_3 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(stack_2_layer_2, self.stack_2_weights['s2_decoder_h1']),
                                                    self.stack_2_biases['s2_decoder_b1'])), keep_prob=self.dropout)

            self.stack_2_layer_4 = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(stack_2_layer_3, self.stack_2_weights['s2_decoder_h2']),
                                                    self.stack_2_biases['s2_decoder_b2'])), keep_prob=self.dropout)

    def _init_decoder_skip_arch(self):
        with tf.variable_scope("decoder") as scope:
            self.layer_5 = tf.nn.dropout(tf.sigmoid(tf.add(self.layer_2, tf.add(tf.matmul(self.layer_3, self.weights['decoder_h1']), self.biases['decoder_b1']))),
                                         keep_prob=self.dropout)

            self.layer_6 = tf.nn.dropout(tf.sigmoid(tf.add(self.layer_1, tf.add(tf.matmul(self.layer_5, self.weights['decoder_h2']), self.biases['decoder_b2']))),
                                         keep_prob=self.dropout)

            self.layer_7 = tf.nn.dropout(tf.sigmoid(tf.add(self.x, tf.add(tf.matmul(self.layer_6, self.weights['decoder_h3']), self.biases['decoder_b3']))),
                                         keep_prob=self.dropout)

    def _init_optimizer(self):
        # Prediction
        if not self.stack_1:
            self.y_pred = self.layer_6
        elif self.stack_1 and not self.stack_2:
            self.y_pred = self.stack_1_layer_4
        elif self.stack_1 and self.stack_2:
            self.y_pred = self.stack_2_layer_4
        # Targets (Labels) are the input data.
        self.loss = tf.reduce_mean(tf.pow(self.y - self.y_pred, 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _init_summary(self):
        loss = tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge([loss])

    def make_inputs(self, data_input, data_targets):
        return {
            self.x: data_input,
            self.y: data_targets,
        }

    def make_inputs_predict(self, data_input):
        return {
            self.x: data_input,
        }
