import datetime
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg') ## for server
import matplotlib.pyplot as plt
##from tqdm import tqdm
import os.path
import time
import sys
import getopt
import pdb
import math
import logging
from skimage.transform import resize
import scipy
import scipy.misc
import matplotlib.image as mpimg
from skimage.util.shape import view_as_windows

from cnn_autoencoder.model import cnn_ae
from cnn_autoencoder.cnn_ae_config import Config as conf

import patch_extraction_module as pem
import data_loading_module as dlm

tf.set_random_seed(123)
np.random.seed(123)

PIXEL_DEPTH = 255
NUM_LABELS = 2
NUM_CHANNELS = 3  # RGB images


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

def extract_data(filename_base, patches_per_image, num_images, patch_size=16,
                 patch_stride=16, train=True):
    """Extract patches from images."""
    dd = np.zeros((num_images*patches_per_image, patch_size, patch_size))
    print(dd.shape)
    for i in range(num_images):
        if train:
            image_fn = filename_base + "satImage_%.3d" % (i+1) + ".png"
        else:
            image_fn = filename_base + "raw_test_" + str(i+1) + "_pixels.png"
        if os.path.isfile(image_fn):
            img = mpimg.imread(image_fn)
        else:
            print('File ' + image_fn + ' does not exist')
        d = view_as_windows(img, window_shape=(patch_size,patch_size),step=patch_stride)
        dd[i*patches_per_image:(i+1)*patches_per_image,:,:] = np.reshape(d, (-1, patch_size, patch_size))
    return dd

def reconstruction(img_data, type):
    """
    Reconstruct single image from flattened array
    Args:
        img_data: flattened image array
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
    # targets = dlm.extract_data(train_data_filename,
    #                            num_images=conf.train_size,
    #                            num_of_transformations=0,
    #                            patch_size=conf.patch_size, # train images are of size 400 for test this needs to be changed
    #                            patch_stride=conf.patch_size, # train images are of size 400 for test this needs to be changed
    #                            border_size=0,
    #                            zero_center=False)
    patches_per_image_train = conf.train_image_size**2 // conf.patch_size**2
    targets = extract_data(train_data_filename, patches_per_image_train, conf.train_size)
    print("Shape of targets: {}".format(targets.shape)) #
    validation = np.copy(targets[:conf.val_size*patches_per_image_train,:,:])
    targets_patch_lvl = np.copy(targets[conf.val_size*conf.patch_size:,:,:])

    del targets # Deleting original data to free space

    print("Training and eval data for CNN DAE")
    train = corrupt(targets_patch_lvl, conf.corruption)
    validation = corrupt(validation, conf.corruption)
    targets = np.copy(targets_patch_lvl)
    del targets_patch_lvl

    print("Shape of training data: {}".format(train.shape)) # (62420, 1, 16, 16)
    print("Shape of targets data: {}".format(targets.shape)) # (62420, 1, 16, 16)
    print("Shape of validation data: {}".format(validation.shape)) # (3125, 1, 16, 16)

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
        validation_writer   = tf.summary.FileWriter("cnn-ae-{}{}-validation-{}".format(conf.log_directory, tag_string, timestamp), graph=tf.get_default_graph())

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

                batch_inputs = train[batch_indices,:,:].reshape((conf.batch_size, conf.patch_size**2))
                batch_targets = targets[batch_indices,:,:].reshape((conf.batch_size, conf.patch_size**2))

                feed_dict = model.make_inputs(batch_inputs, batch_targets)

                _, train_summary = sess.run([model.optimizer, model.summary_op], feed_dict)
                train_writer.add_summary(train_summary, global_step)

                global_step += 1
                batch_index += 1

        saver.save(sess, os.path.join(train_logfolderPath, "cnn-ae-{}-{}-ep{}-final.ckpt".format(tag_string, timestamp, conf.num_epochs)))
        print("Done with training for {} epochs".format(conf.num_epochs))

        if conf.visualise_training:
            print("Visualising encoder results and true images from train set")
            data_eval_fd = validation.reshape((conf.val_size*patches_per_image_train, conf.patch_size**2))
            feed_dict = model.make_inputs_predict(data_eval_fd)
            encode_decode = sess.run(model.y_pred, feed_dict=feed_dict) ## predictions from model are [batch_size, dim, dim, n_channels] i.e. (3125, 16, 16, 1)
            print("shape of predictions: {}".format(encode_decode.shape))
            # Compare original images with their reconstructions
            f, a = plt.subplots(2, conf.examples_to_show, figsize=(conf.examples_to_show, 5))
            for i in range(conf.examples_to_show):
                val = reconstruction(validation[i*patches_per_image_train:((i+1)*patches_per_image_train),:,:], type='train')
                pred = reconstruction(encode_decode[i*patches_per_image_train:((i+1)*patches_per_image_train),:,:,0], type = 'train')
                a[0][i].imshow(val)
                im = a[1][i].imshow(pred)
            plt.colorbar(im)
            plt.savefig('./cnn_autoencoder_eval_{}.png'.format(tag))

        print("Deleting train and targets objects")
        del train
        del targets

        if conf.run_on_test_set:
            print("DAE on the predictions")
            prediction_test_dir = "../results/CNN_Output/test/high_res_raw/"
            output_path_raw = "../results/CNN_Autoencoder_Output/raw/"
            if not os.path.isdir(prediction_test_dir):
                raise ValueError('no CNN data to run denoising autoencoder on')

            print("Loading test set")
            # test = dlm.extract_data(prediction_test_dir,
            #                         num_images=conf.test_size,
            #                         num_of_transformations=0,
            #                         patch_size=conf.patch_size, # train images are of size 400 for test this needs to be changed
            #                         patch_stride=conf.patch_size, # train images are of size 400 for test this needs to be changed
            #                         border_size=0,
            #                         zero_center=False,
            #                         autoencoder=True) ## uses different path to load data
            patches_per_image_test = conf.test_image_size**2 // conf.patch_size**2
            test = extract_data(prediction_test_dir, patches_per_image_test, conf.test_size, train=False)
            print("Shape of test set: {}".format(test.shape)) ## (72200, 1, 16, 16)

            # feeing in one image at a time
            predictions = []
            inputs = test.reshape((test.shape[0], conf.patch_size**2))
            for i in range(conf.test_size):
                batch_inputs = inputs[i*patches_per_image_test:((i+1)*patches_per_image_test),:]
                feed_dict = model.make_inputs_predict(batch_inputs)
                prediction = sess.run(model.y_pred, feed_dict) ## numpy array (50, 76, 76, 1)
                predictions.append(prediction)

            # Save outputs to disk
            for i in range(conf.test_size):
                print("Test img: " + str(i+1))
                img_name = "cnn_ae_test_" + str(i+1)
                output_path = "../results/CNN_Autoencoder_Output/tmp/" + img_name
                print(predictions[i].shape) # (1444, 16, 16, 1) 
                prediction = reconstruction(predictions[i][:,:,:,0], type='test')
                scipy.misc.imsave(output_path + ".png", prediction)

            f, a = plt.subplots(2, conf.examples_to_show, figsize=(conf.examples_to_show, 5))
            for i in range(conf.examples_to_show):
                t = reconstruction(test[i*patches_per_image_test:((i+1)*patches_per_image_test),:,:], type='test')
                pred = reconstruction(predictions[i][:,:,:,0], type='test')
                a[0][i].imshow(t)
                im = a[1][i].imshow(pred)
            plt.colorbar(im)
            plt.savefig('./cnn_autoencoder_prediction_{}.png'.format(tag))

            print("Finished saving cnn autoencoder outputs to disk")

if __name__ == "__main__":
    #logging.basicConfig(filename='autoencoder.log', level=logging.DEBUG)
    mainFunc(sys.argv[1:])
