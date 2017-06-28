""" CNN De-noising Auto Encoder Example.
ref: https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py
"""
import math
import tensorflow as tf
import numpy as np

class cnn_ae():
    """CNN Autoencoder class"""
    def __init__(self,
                 n_input, ## dim of the inputs
                 n_filters=[1, 10, 10, 10],
                 filter_sizes=[5, 5, 3, 3],
                 learning_rate=0.0001):

        # Network Parameters
        self.n_input = n_input
        self.n_filters = n_filters
        self.layers = len(n_filters)-1
        self.filter_sizes = filter_sizes
        self.learning_rate = learning_rate

        self._make_graph()

    def _make_graph(self):
        tf.reset_default_graph()

        self._init_placeholders()

        self._init_encoder()
        self._init_decoder()

        self._init_optimizer()
        self._init_summary()

    def _init_placeholders(self):
        self.x = tf.placeholder(shape=(None, self.n_input),
                                dtype=tf.float32,
                                name='encoder_inputs',
                                )

        # convert to 4d tensor
        x_dim = np.sqrt(self.x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(self.x, [-1, x_dim, x_dim, self.n_filters[0]])
        self.current_input = x_tensor

        self.y = tf.placeholder(shape=(None, self.n_input),
                                dtype=tf.float32,
                                name='targets',
                                )

        # convert to 4d tensor
        y_dim = np.sqrt(self.y.get_shape().as_list()[1])
        if y_dim != int(y_dim):
            raise ValueError('Unsupported input dimensions')
        y_dim = int(y_dim)
        y_tensor = tf.reshape(self.y, [-1, x_dim, x_dim, self.n_filters[0]])
        self.target = y_tensor

    # Building the encoder
    def _init_encoder(self):
        ## Uniform(-sqrt(3), sqrt(3)) has variance=1.
        sqrt3 = math.sqrt(3)
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
        self.encoder = []
        self.shapes = []
        print(self.current_input.get_shape())
        for layer_i, n_output in enumerate(self.n_filters[1:]):
            n_input = self.current_input.get_shape().as_list()[3]
            self.shapes.append(self.current_input.get_shape().as_list())
            W = tf.get_variable(name="encoder_w"+str(layer_i),
                                shape=[self.filter_sizes[layer_i], self.filter_sizes[layer_i], n_input, n_output],
                                initializer=initializer,
                                dtype=tf.float32)
            b = tf.get_variable(name="encoder_b"+str(layer_i),
                                shape=[n_output],
                                initializer=initializer,
                                dtype=tf.float32)
            self.encoder.append(W)
            output = tf.add(tf.nn.conv2d(self.current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b)
            print("shape of encoder outputs: {}".format(output.get_shape()))
            self.current_input = tf.sigmoid(output)

    def _init_decoder(self):
        ## Uniform(-sqrt(3), sqrt(3)) has variance=1.
        sqrt3 = math.sqrt(3)
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
        self.encoder.reverse()
        self.shapes.reverse()
        for layer_i, shape in enumerate(self.shapes):
            W = self.encoder[layer_i]
            b = tf.get_variable(name="decoder_b"+str(layer_i),
                                shape=[W.get_shape().as_list()[2]],
                                initializer=initializer,
                                dtype=tf.float32)
            output = tf.add(
                tf.nn.conv2d_transpose(
                    self.current_input, W,
                    tf.stack([tf.shape(self.x)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), b)
            print("shape of decoder outputs: {}".format(output.get_shape()))
            self.current_input = tf.sigmoid(output)

    def _init_optimizer(self):
        # Prediction
        self.y_pred = self.current_input
        # Targets (Labels) are the input data.
        self.loss = tf.reduce_mean(tf.pow(self.target - self.y_pred, 2))
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
