import datetime
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg') ## for server
import matplotlib.pyplot as plt
import os.path
import time
import sys
import getopt
import pdb
import math
import logging
import scipy
import scipy.misc
import matplotlib.image as mpimg
from skimage.util.shape import view_as_windows
from skimage.transform import resize
from sklearn.feature_extraction import image

from cnn_autoencoder.model import cnn_ae
from cnn_autoencoder.cnn_ae_config import Config as conf

tf.set_random_seed(123)
np.random.seed(123)

def corrupt(data, nu, type='salt_and_pepper'):
    """
    Corrupts the data for inputing into the de-noising autoencoder

    Args:
        data: numpy array of size (num_points, 1, img_size, img_size)
        nu: corruption level
    Returns:
        numpy array of size (num_points, 1, img_size, img_size)
    """
    if type == 'salt_and_pepper':
        img_max = np.ones(data.shape, dtype=bool)
        tmp = np.copy(data)
        img_max[data <= 0.5] = False
        img_min = np.logical_not(img_max)
        idx = np.random.choice(a = [True, False], size=data.shape, p=[nu, 1-nu])
        tmp[np.logical_and(img_max, idx)] = 0
        tmp[np.logical_and(img_min, idx)] = 1
    return tmp

def extract_data(filename_base, num_images, train=True):
    """Data from disk"""
    if train:
        dd = np.zeros((num_images, conf.train_image_size, conf.train_image_size))
    else:
        dd = np.zeros((num_images, conf.test_image_size, conf.test_image_size))
    for i in range(num_images):
        if train:
            image_fn = filename_base + "satImage_%.3d" % (i+1) + ".png"
        else:
            image_fn = filename_base + "raw_test_" + str(i+1) + "_pixels.png"
        if os.path.isfile(image_fn):
            img = mpimg.imread(image_fn)
        else:
            print('File ' + image_fn + ' does not exist')
        dd[i,:,:] = img
    return dd

def resize_train_cnn_output_lvl(data):
    """
    Resize data so that individual pixels are 8x8 for noising
    Args:
        data: numpy array of training data
    Returns:
        numpy array of size 50x50
    """
    new_size = conf.train_image_size // conf.cnn_pred_size # 50
    dd = np.zeros((data.shape[0], new_size, new_size))
    for i in range(data.shape[0]):
        dd[i,:,:] = resize(data[i,:,:], (new_size, new_size))
    return dd

def resize_train_lvl(data):
    """
    Resize data so that individual pixels are 8x8 for noising
    Args:
        data: numpy array of training data at cnn output lvl i.e. 50x50
    Returns:
        numpy array of size 400x400
    """
    dd = np.zeros((data.shape[0], 400, 400))
    for i in range(data.shape[0]):
        dd[i,:,:] = resize(data[i,:,:], (400, 400))
    return dd

def reconstruction(img_data, type):
    """
    Reconstruct single image from flattened array
    Args:
        img_data: 3d array (num patchs x patch size x patch size)
        type: str train / test
    Returns:
        recontructed image
    """
    if type == "train":
        size = conf.train_image_size
    elif type == "test":
        size = conf.test_image_size
    else:
        ValueError('train or test plz')
    reconstruction = np.zeros((size,size))
    r = size // conf.patch_size
    idx = 0
    for i in range(int(r)):
        for j in range(int(r)):
            reconstruction[i*conf.patch_size:(i+1)*conf.patch_size,j*conf.patch_size:(j+1)*conf.patch_size] =  img_data[idx,:,:]
            idx += 1
    return reconstruction

def patchify(data, train=True):
    """
    Converts data into patches of feeding into the CNN
    Train used sliding window of ste size 2
    Test uses sliding window where the patch size is the stride
    Args:
        data: numpy array of shape (n_points, dim_x, dim_y)
        train: bool
    Returns:
        numpy array of size (n, patch_size, patch_size)
    """

    patches = []
    for i in range(data.shape[0]):
        if train:
            patches.append(image.extract_patches(data[i,:,:], conf.patch_size, extraction_step=1))
            patches.append(image.extract_patches(np.rot90(data[i,:,:]), conf.patch_size, extraction_step=1))
        else:
            patches.append(view_as_windows(data[i,:,:], window_shape=(conf.patch_size,conf.patch_size),step=conf.patch_size))
    dd = np.stack(patches).reshape(-1, conf.patch_size, conf.patch_size)
    return dd

def mainFunc(argv):
    def printUsage():
        print('main.py -n <num_cores> -t <tag>')
        print('num_cores = Number of cores requested from the cluster. Set to -1 to leave unset')
        print('tag = optional tag or name to distinguish the runs, e.g. \'bidirect3layers\' ')

    num_cores = -1
    tag = None
    # Command line argument handling
    try:
        opts, args = getopt.getopt(argv,"n:t:",["num_cores=", "tag="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        elif opt in ("-n", "--num_cores"):
            num_cores = int(arg)
        elif opt in ("-t", "--tag"):
            tag = arg

    print("Executing autoencoder with {} CPU cores".format(num_cores))
    if num_cores != -1:
        # We set the op_parallelism_threads in the ConfigProto and pass it to the TensorFlow session
        configProto = tf.ConfigProto(inter_op_parallelism_threads=num_cores,
                                     intra_op_parallelism_threads=num_cores)
    else:
        configProto = tf.ConfigProto()

    print("loading ground truth data")
    train_data_filename = "../data/training/groundtruth/"
    targets = extract_data(train_data_filename, num_images=conf.train_size, train=True)
    print("Shape of targets: {}".format(targets.shape)) # (100, 400, 400)
    validation = np.copy(targets[:conf.val_size,:,:])
    targets = np.copy(targets[conf.val_size:,:,:])

    print("Re-sizing targets and validation")
    targets = resize_train_cnn_output_lvl(targets)
    validation = resize_train_cnn_output_lvl(validation)
    print("new size of targets: {}".format(targets.shape)) # (95, 50, 50)
    print("new size of validation: {}".format(validation.shape)) # (5, 50, 50)

    print("Adding noise to training data")
    train = corrupt(targets, conf.corruption)
    validation = corrupt(validation, conf.corruption)

    print("Re-sizing validation back to train lvl")
    validation = resize_train_lvl(validation)

    print("Patchifying data for network")
    train = patchify(train, train=True)
    targets = patchify(targets, train=True)
    validation = patchify(validation, train=False)

    print("Shape of training data: {}".format(train.shape)) # (232750, 16, 16)
    print("Shape of targets data: {}".format(targets.shape)) # (232750, 16, 16)
    print("Shape of validation data: {}".format(validation.shape)) # (45, 16, 16)

    print("Initializing CNN denoising autoencoder")
    model = cnn_ae(conf.patch_size**2, ## dim of the inputs
                   n_filters=[1, 16, 32, 64],
                   filter_sizes=[7, 5, 3, 3],
                   learning_rate=0.005)

    print("Starting TensorFlow session")
    with tf.Session(config=configProto) as sess:
        start = time.time()
        global_step = 1

        saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=2)

        # Init Tensorboard summaries. This will save Tensorboard information into a different folder at each run.
        timestamp = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        tag_string = ""
        if tag is not None:
            tag_string = tag
        train_logfolderPath = os.path.join(conf.log_directory, "cnn-ae-{}-training-{}".format(tag_string, timestamp))
        train_writer        = tf.summary.FileWriter(train_logfolderPath, graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())

        sess.graph.finalize()

        print("Starting training")
        for i in range(conf.num_epochs):
            print("Training epoch {}".format(i))
            print("Time elapsed:    %.3fs" % (time.time() - start))

            n = train.shape[0]
            perm_idx = np.random.permutation(n)
            batch_index = 1
            for step in range(int(n / conf.batch_size)):
                offset = (batch_index*conf.batch_size) % (n - conf.batch_size)
                batch_indices = perm_idx[offset:(offset + conf.batch_size)]

                batch_inputs = train[batch_indices,:].reshape((conf.batch_size, conf.patch_size**2))
                batch_targets = targets[batch_indices,:].reshape((conf.batch_size, conf.patch_size**2))

                feed_dict = model.make_inputs(batch_inputs, batch_targets)

                _, train_summary = sess.run([model.optimizer, model.summary_op], feed_dict)
                train_writer.add_summary(train_summary, global_step)

                global_step += 1
                batch_index += 1

        saver.save(sess, os.path.join(train_logfolderPath, "cnn-ae-{}-{}-ep{}-final.ckpt".format(tag_string, timestamp, conf.num_epochs)))
        print("Done with training for {} epochs".format(conf.num_epochs))

        if conf.visualise_training:
            print("Visualising encoder results and true images from train set")
            patches_per_image_train = conf.train_image_size**2 // conf.patch_size**2
            data_eval_fd = validation.reshape((conf.val_size*patches_per_image_train, conf.patch_size**2))
            feed_dict = model.make_inputs_predict(data_eval_fd)
            encode_decode = sess.run(model.y_pred, feed_dict=feed_dict) ## predictions from model are [batch_size, dim, dim, n_channels] i.e. (3125, 16, 16, 1)
            print("shape of predictions: {}".format(encode_decode.shape))
            # Compare original images with their reconstructions
            f, a = plt.subplots(2, conf.examples_to_show, figsize=(conf.examples_to_show, 5))
            for i in range(conf.examples_to_show):
                val = reconstruction(validation[i*patches_per_image_train:((i+1)*patches_per_image_train),:,:], type='train')
                pred = reconstruction(encode_decode[i*patches_per_image_train:((i+1)*patches_per_image_train),:,:,0], type = 'train')
                a[0][i].imshow(val, cmap='gray', interpolation='none')
                a[1][i].imshow(pred, cmap='gray', interpolation='none')
                a[0][i].get_xaxis().set_visible(False)
                a[0][i].get_yaxis().set_visible(False)
                a[1][i].get_xaxis().set_visible(False)
                a[1][i].get_yaxis().set_visible(False)
            plt.gray()
            plt.savefig('./cnn_autoencoder_eval_{}.png'.format(tag))

        # Deleting train and targets objects
        del train
        del targets

        if conf.run_on_test_set:
            print("Running the Convolutional Denoising Autoencoder on the predictions")
            prediction_test_dir = "../results/CNN_Output/test/high_res_raw/"
            if not os.path.isdir(prediction_test_dir):
                raise ValueError('no CNN data to run Convolutional Denoising Autoencoder on')

            print("Loading test set")
            patches_per_image_test = conf.test_image_size**2 // conf.patch_size**2
            test = extract_data(prediction_test_dir, conf.test_size, train=False) # 50 x 608 x 608
            inputs = patchify(test, train=False)
            print("Shape of test set: {}".format(inputs.shape)) ##

            # feeding in one image at a time in the convolutional autoencoder for prediction
            # where the batch size is the number of patches per test image
            predictions = []
            for i in range(conf.test_size):
                batch_inputs = inputs[i*patches_per_image_test:((i+1)*patches_per_image_test),:,:].reshape((patches_per_image_test, conf.patch_size**2))
                feed_dict = model.make_inputs_predict(batch_inputs)
                prediction = sess.run(model.y_pred, feed_dict) ## numpy array (50, 76, 76, 1)
                predictions.append(prediction)

            # Save outputs to disk
            for i in range(conf.test_size):
                print("Test img: " + str(i+1))
                img_name = "cnn_ae_test_" + str(i+1)
                output_path = "../results/CNN_Autoencoder_Output/high_res_raw/"
                if not os.path.isdir(output_path):
                    raise ValueError('no CNN data to run Convolutional Denoising Autoencoder on')
                print(predictions[i].shape) # (1444, 16, 16, 1)
                prediction = reconstruction(predictions[i][:,:,:,0], type='test')
                scipy.misc.imsave(output_path + img_name + ".png", prediction)

            f, a = plt.subplots(2, conf.examples_to_show, figsize=(conf.examples_to_show, 5))
            for i in range(conf.examples_to_show):
                t = reconstruction(inputs[i*patches_per_image_test:((i+1)*patches_per_image_test),:,:], type='test')
                pred = reconstruction(predictions[i][:,:,:,0], type='test')
                a[0][i].imshow(t, cmap='gray', interpolation='none')
                a[1][i].imshow(pred, cmap='gray', interpolation='none')
                a[0][i].get_xaxis().set_visible(False)
                a[0][i].get_yaxis().set_visible(False)
                a[1][i].get_xaxis().set_visible(False)
                a[1][i].get_yaxis().set_visible(False)
            plt.gray()
            plt.savefig('./cnn_autoencoder_prediction_{}.png'.format(tag))

            print("Finished saving cnn autoencoder outputs to disk")

if __name__ == "__main__":
    #logging.basicConfig(filename='autoencoder.log', level=logging.DEBUG)
    mainFunc(sys.argv[1:])
