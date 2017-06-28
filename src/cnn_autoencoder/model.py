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

class cnn_ae_ethan():
    """Simple 2 layer model"""
    def __init__(self,
                 w, ## dim of the inputs,
                 learning_rate=0.005):

        self.w = w
        self.learning_rate = learning_rate
        self._make_graph()

    def _make_graph(self):
        tf.reset_default_graph()

        self._init_placeholders()
        self._build_graph()
        self._init_optimizer()
        self._init_summary()

    def _init_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape = [None, self.w * self.w])
        self.x_noise = tf.placeholder(tf.float32, shape = [None, self.w * self.w])

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
        # Reshaping is important!!!
        self.x_origin = tf.reshape(self.x, [-1, self.w, self.w, 1])
        x_origin_noise = tf.reshape(self.x_noise, [-1, self.w, self.w, 1])

        W_e_conv1 = self.weight_variable([5, 5, 1, 16], "w_e_conv1")
        b_e_conv1 = self.bias_variable([16], "b_e_conv1")
        h_e_conv1 = tf.nn.relu(tf.add(self.conv2d(x_origin_noise, W_e_conv1), b_e_conv1))

        W_e_conv2 = self.weight_variable([5, 5, 16, 32], "w_e_conv2")
        b_e_conv2 = self.bias_variable([32], "b_e_conv2")
        h_e_conv2 = tf.nn.relu(tf.add(self.conv2d(h_e_conv1, W_e_conv2), b_e_conv2))

        code_layer = h_e_conv2
        print("code layer shape : %s" % h_e_conv2.get_shape())

        W_d_conv1 = self.weight_variable([5, 5, 16, 32], "w_d_conv1")
        b_d_conv1 = self.bias_variable([1], "b_d_conv1")
        output_shape_d_conv1 = tf.stack([tf.shape(self.x)[0], int(self.w/2), int(self.w/2), 16])
        h_d_conv1 = tf.nn.relu(self.deconv2d(h_e_conv2, W_d_conv1, output_shape_d_conv1))

        W_d_conv2 = self.weight_variable([5, 5, 1, 16], "w_d_conv2")
        b_d_conv2 = self.bias_variable([16], "b_d_conv2")
        output_shape_d_conv2 = tf.stack([tf.shape(self.x)[0], self.w, self.w, 1])
        h_d_conv2 = tf.nn.relu(self.deconv2d(h_d_conv1, W_d_conv2, output_shape_d_conv2))

        self.y_pred = h_d_conv2
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
