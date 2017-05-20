"""
Baseline for CIL project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import pdb
from skimage.transform import rotate
import code
import tensorflow.python.platform
import numpy as np
import tensorflow as tf

NUM_CHANNELS = 3 # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
TEST_SIZE = 50
VALIDATION_SIZE = 5  # Size of the validation set. Not used
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16 # 64
NUM_EPOCHS = 5
RESTORE_MODEL = False # If True, restore existing model instead of training a new one
RECORDING_STEP = 1000
ROTATE_IMAGES = False

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE= 62 ## Training images are 400 x 400 x 3, Testing are 608 x 608 x 3

tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS

def img_crop(im, w, h, aerial_image):
    """
    Extracts patches from a given image
    Args:
        im: image
        w: width
        h: height
    Returns:
        list with patches of images
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3 ## 3 channels RBG
    ## creating padding in the image if there is not a whole number of patches which fit on the image
    ## creating padding for bottom of image
    rw = imgwidth % w
    pad_width = w - rw
    if rw != 0:
        if aerial_image:
            if is_2d:
                ## append means to image channels
                tmp = np.full((pad_width, imgheight), np.mean(im), dtype=float)
                pdb.set_trace()
                im_padded = np.concatenate((im,tmp), axis=0)
            else:
                ## append mean to width of image
                tmp_r = np.full((pad_width, imgheight), np.mean(im[:,:,0]), dtype=float)
                tmp_b = np.full((pad_width, imgheight), np.mean(im[:,:,1]), dtype=float)
                tmp_g = np.full((pad_width, imgheight), np.mean(im[:,:,2]), dtype=float)
                tmp = np.stack((tmp_r, tmp_b, tmp_g), axis=2) ## axis=2 creates new dim
                assert tmp.shape == (pad_width, imgheight, 3), 'width padding not the correct shape'
                ##pdb.set_trace()
                im_padded = np.concatenate((im, tmp), axis=0)
        else:
            ## gt images are of size (w,h) no RBG
            tmp = np.full((pad_width, imgheight), 0, dtype=int)
            im_padded = np.concatenate((im,tmp), axis=0)

    ## creating padding for the right of the image
    imgwidth_new = im_padded.shape[0]
    rh = imgheight % h
    pad_height = h - rh
    if rh != 0:
        if aerial_image:
            if is_2d:
                tmp = np.full((imgwidth_new, pad_height), np.mean(im), dtype=int)
                im_padded = np.concatenate((im_padded,tmp), axis=1)
            else:
                ## append mean to width of image
                tmp_r = np.full((imgwidth_new, pad_height), np.mean(im[:,:,0]), dtype=float)
                tmp_b = np.full((imgwidth_new, pad_height), np.mean(im[:,:,1]), dtype=float)
                tmp_g = np.full((imgwidth_new, pad_height), np.mean(im[:,:,2]), dtype=float)
                tmp = np.stack((tmp_r, tmp_b, tmp_g), axis=2)
                assert tmp.shape == (imgwidth_new, pad_height, 3), 'height padding not the correct shape'
                im_padded = np.concatenate((im_padded, tmp), axis=1) ## axis = 1 to concatenate along cols
                ##Image.fromarray(img_float_to_uint8(im_padded)).save("padded.png")
                ##pdb.set_trace()
        else:
            ## gt images are of size (w,h) no RBG
            tmp = np.full((imgwidth_new, pad_height), 0, dtype=int)
            assert tmp.shape == (imgwidth_new, pad_height), 'height padding not the correct shape'
            im_padded = np.concatenate((im_padded,tmp), axis=1)
            ##Image.fromarray(img_float_to_uint8(im_padded)).save("padded_gt.png")
            ##pdb.set_trace()

    ## overwriting original height
    imgheight_new = im_padded.shape[1]
    ##pdb.set_trace()
    assert imgwidth_new % w == 0 and imgheight_new % h == 0, 'New img dimensions are not wholly covered by patches'
    for i in range(0,imgheight_new,h):
        for j in range(0,imgwidth_new,w):
            if is_2d:
                im_patch = im_padded[j:j+w, i:i+h]
            else:
                im_patch = im_padded[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            if ROTATE_IMAGES:
                pass
            imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    if ROTATE_IMAGES:
        imgs
    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]

    ## Dividing each image up into patches of size [IMG_PATCH_SIZE x IMG_PATCH_SIZE]
    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, aerial_image=True) for i in range(num_images)] ## list of list
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    ##pdb.set_trace()

    return np.asarray(data)

# Assign a label to a patch v
def value_to_class(v):
    """
    Args:
        v: patch
    Returns:
        one hot encoding of the label of the patch
    """
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [1, 0] ## road
        ## return [0, 1]
    else:
        return [0, 1] ## not road
        ## return [1, 0]

# Extract label images
def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print ('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    ## cropping images to [IMG_PATCH_SIZE, IMG_PATCH_SIZE]
    ## pdb.set_trace()
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE, aerial_image=False) for i in range(num_images)] ## list of lists
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))]) ## converting to an array
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Args:
        imgwidth: the width of the image
        imgheight: the height of the image
        w: the width of the patch of image
        h: the height of the patch of the image
        labels: tensor the predictions
    Returns:
        numpy array of the size [imgwidth, imgheight] with 0s and 1s indicating not road not road
        respectively
    """
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def data2img(data):
    """
    Converts 2d numpy data to image to RGB image ready for viewing.
    Args:
        data is a 2d numpy array
    Returns:
        img is an image object which may be saved
    """
    w = data.shape[0]
    h = data.shape[1]
    assert len(data.shape) < 3, 'data needs to be 2d'
    img_3c = np.zeros((w, h, 3), dtype=np.uint8)
    img8 = img_float_to_uint8(data)
    img_3c[:,:,0] = img8
    img_3c[:,:,1] = img8
    img_3c[:,:,2] = img8
    return img_3c

def concatenate_images(img, gt_img):
    """
    Concatenates an image with its ground truth for easy visulisation
    """
    nChannels = len(gt_img.shape)
    ##w = gt_img.shape[0]
    ##h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = data2img(gt_img)
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    """
    Overlays the prediction on top of the image
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def main(argv=None):  # pylint: disable=unused-argument

    data_dir = 'training/'
    train_data_dir = data_dir + 'images/'
    train_labels_dir = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_dir, TRAINING_SIZE)
    train_labels = extract_labels(train_labels_dir, TRAINING_SIZE)

    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print ('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print("length of new_indices: {}".format(len(new_indices)))
    print("shape of training data: {}".format(train_data.shape))
    train_data = train_data[new_indices,:,:,:]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print ('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx = 0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value*PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image
    def get_prediction(img):
        data = np.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, aerial_image=False))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
        return img_prediction

    def _get_prediction(fn):
        """
        Wrapper around the get_prediction function
        """
        print("prediction for image: {}".format(fn))
        img = mpimg.imread(fn)
        data = get_prediction(img)
        ## Convert to RGB image
        nChannels = len(data.shape)
        if nChannels == 3:
            return data
        else:
            return data2img(data)

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_with_groundtruth(directory, image_idx, training=True):
        """
        Args:
            directory: str
            image_idx: int
            training: bool
        Returns:
            Prediction for the image input, prediction is concatenated with original
        """
        if training:
            imageid = "satImage_%.3d" % image_idx
        else:
            imageid = "test_" + str(image_idx)
        image_filename = directory + imageid + ".png"
        print("prediction for image: {}".format(image_filename))
        img = mpimg.imread(image_filename)
        img_prediction = get_prediction(img)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(directory, image_idx, training=True):
        """
        Args:
            directory: str
            image_idx: int
            training: bool
        Returns:
            Prediction for the image input, prediction is overlayed on original
        """
        if training:
            imageid = "satImage_%.3d" % image_idx
        else:
            imageid = "test_" + str(image_idx)
        image_filename = directory + imageid + ".png"
        print("prediction for image: {}".format(image_filename))
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # Uncomment these lines to check the size of each layer
        print 'data ' + str(data.get_shape())
        print 'conv ' + str(conv.get_shape())
        print 'relu ' + str(relu.get_shape())
        print 'pool ' + str(pool.get_shape())
        print 'pool2 ' + str(pool2.get_shape())


        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        pdb.set_trace()
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        #if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        if train == True:
            summary_id = '_0'
            s_data = get_image_summary(data)
            filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
            filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool)
            s_conv2 = get_image_summary(conv2)
            filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool2)

        return out

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True) # BATCH_SIZE*NUM_LABELS
    # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels_node))
    tf.summary.scalar('loss', loss)

    all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases]
    all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases', 'fc2_weights', 'fc2_biases']
    all_grads_node = tf.gradients(loss, all_params_node)
    all_grad_norms_node = []
    for i in range(0, len(all_grads_node)):
        norm_grad_i = tf.global_norm([all_grads_node[i]])
        all_grad_norms_node.append(norm_grad_i)
        tf.summary.scalar(all_params_names[i], norm_grad_i)

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.0).minimize(loss,
                                                         global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:
        if RESTORE_MODEL:
            # Restore variables from disk.
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                    graph_def=s.graph_def)
            print ('Initialized!')
            # Loop through training steps.
            print ('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

            training_indices = range(train_size)

            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = np.random.permutation(training_indices)

                for step in range (int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a numpy array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_labels_node: batch_labels}

                    if step % RECORDING_STEP == 0:

                        summary_str, _, l, lr, predictions = s.run(
                            [summary_op, optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)
                        #summary_str = s.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        # print_predictions(predictions, batch_labels)

                        print ('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
                        print ('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        print ('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                     batch_labels))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                # Save the variables to disk.
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        if not RESTORE_MODEL:
            print ("Running prediction on training set")
            prediction_training_dir = "predictions_training/"
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)
            for i in range(1, TRAINING_SIZE+1):
                pimg = get_prediction_with_groundtruth(train_data_dir, i)
                Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
                oimg = get_prediction_with_overlay(train_data_dir, i)
                oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")
        else:
            print("Running prediction on test set")
            test_dir = "test_set_images/"
            for i in range(1, TEST_SIZE+1):
                test_data_filename = test_dir + "test_" + str(i) + "/" ## the rest of the string is added in the prediction functions below
                pimg = get_prediction_with_groundtruth(test_data_filename, i, training=False)
                Image.fromarray(pimg).save(test_data_filename + "prediction_" + str(i) + ".png")
                oimg = get_prediction_with_overlay(test_data_filename, i, training=False)
                oimg.save(test_data_filename + "overlay_" + str(i) + ".png")
                img  = _get_prediction(test_data_filename +  "test_" + str(i) + ".png")
                Image.fromarray(img).save(test_data_filename + "final_" + str(i) + ".png")

if __name__ == '__main__':
    tf.app.run()
