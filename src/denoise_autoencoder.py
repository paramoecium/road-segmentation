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

from autoencoder.model import ae
from autoencoder.ae_config import Config as conf

import patch_extraction_module as pem
import data_loading_module as dlm
import constants as const

tf.set_random_seed(123)
np.random.seed(123)

ROOT_DIR = "../"
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
    train_data_filename = ROOT_DIR + "data/training/groundtruth/"
    targets = dlm.extract_data(train_data_filename,
                               num_images=conf.train_size,
                               num_of_transformations=0,
                               patch_size=conf.train_image_size, # train images are of size 400 for test this needs to be changed
                               patch_stride=conf.train_image_size, # train images are of size 400 for test this needs to be changed
                               border_size=0,
                               zero_center=False)

    # The output of the CNN is predictions of size 608x608, the prediction is made on patches of size 8x8.
    # Hence the denoising with be on made on images of size 76x76. Hence the training of the denoising autoencoder
    # Is made on the training data (size 400x400) resized (spline interpolation) to 76x76
    print("Resizing ground truth images so that patches from CNN are now pixels")
    targets_patch_lvl = np.zeros((targets.shape[0], conf.test_image_resize, conf.test_image_resize))
    for i in range(targets.shape[0]):
        targets_patch_lvl[i,:,:] = resize(targets[i,0,:,:],
                                          (conf.test_image_resize, conf.test_image_resize),
                                          order=0, preserve_range=True)

    print("New shape of each image: {}".format(targets_patch_lvl.shape)) ## (5, 50, 50)

    del targets # Deleting original data to free space

    print("Initializing model")
    print("Input size: {}".format(int(conf.test_image_resize*conf.test_image_resize)))
    print("H1 size: {}".format(int(conf.test_image_resize*conf.test_image_resize/conf.ae_step)))
    print("H2 size: {}".format(int(conf.test_image_resize*conf.test_image_resize/conf.ae_step/conf.ae_step)))
    print("H3 size: {}".format(int(conf.test_image_resize*conf.test_image_resize/conf.ae_step/conf.ae_step/conf.ae_step)))
    ##print("H4 size: {}".format(int(conf.test_image_resize*conf.test_image_resize/conf.ae_step/conf.ae_step/conf.ae_step/conf.ae_step)))

    model = ae(n_input=int(conf.test_image_resize*conf.test_image_resize),
               n_hidden_1=int(conf.test_image_resize*conf.test_image_resize/conf.ae_step),
               n_hidden_2=int(conf.test_image_resize*conf.test_image_resize/conf.ae_step/conf.ae_step),
               n_hidden_3=int(conf.test_image_resize*conf.test_image_resize/conf.ae_step/conf.ae_step/conf.ae_step),
               ##n_hidden_4=int(conf.test_image_resize*conf.test_image_resize/conf.ae_step/conf.ae_step/conf.ae_step/conf.ae_step),
               learning_rate=conf.learning_rate,
               dropout=conf.dropout_train)

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
        train_logfolderPath = os.path.join(conf.log_directory, "{}-training-{}".format(tag_string, timestamp))
        train_writer        = tf.summary.FileWriter(train_logfolderPath, graph=tf.get_default_graph())
        validation_writer   = tf.summary.FileWriter("{}{}-validation-{}".format(conf.log_directory, tag_string, timestamp), graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())

        sess.graph.finalize()

        print("Starting training")
        for i in range(conf.num_epochs):
            if i % 100 == 0:
                print("Training epoch {}".format(i))
                #logging.info("Training epoch {}".format(i))
                print("Time elapsed:    %.3fs" % (time.time() - start))
                #logging.info("Time elapsed:    %.3fs" % (time.time() - start))

            ## corrupting the ground truth labels
            train = corrupt(targets_patch_lvl, float(np.random.choice(a = [0.01, 0.03], size=1, p=[0.5, 0.5])))

            perm_idx = np.random.permutation(conf.train_size)
            batch_index = 1
            for step in range(int(conf.train_size / conf.batch_size)):
                offset = (batch_index*conf.batch_size) % (conf.train_size - conf.batch_size)
                batch_indices = perm_idx[offset:(offset + conf.batch_size)]

                batch_inputs = train[batch_indices,:,:].reshape((conf.batch_size, conf.test_image_resize**2))
                batch_targets = targets_patch_lvl[batch_indices,:,:].reshape((conf.batch_size, conf.test_image_resize**2))

                ##print("shape of batch inputs: {0} and outputs: {1}".format(batch_inputs.shape, batch_targets.shape))

                ##pdb.set_trace()
                feed_dict = model.make_inputs(batch_inputs, batch_targets)
                if global_step % conf.validation_summary_frequency == 0:
                    pass
                else:
                    _, train_summary = sess.run([model.optimizer, model.summary_op], feed_dict)
                    train_writer.add_summary(train_summary, global_step)

                if global_step % conf.checkpoint_frequency == 0:
                    saver.save(sess, os.path.join(train_logfolderPath, "{}-{}-ep{}.ckpt".format(tag_string, timestamp, i)), global_step=global_step)
                global_step += 1
                batch_index += 1

        saver.save(sess, os.path.join(train_logfolderPath, "{}-{}-ep{}-final.ckpt".format(tag_string, timestamp, conf.num_epochs)))
        print("Done with training for {} epochs".format(conf.num_epochs))

        print("Visualising encoder results and true images from eval set")
        # Applying encode and decode over test set
        # One batch for eval
        data_eval = train[batch_indices,:,:]
        data_eval_fd = data_eval.reshape((conf.batch_size, conf.test_image_resize**2))
        targets_eval = targets_patch_lvl[batch_indices,:,:]
        targets_eval_fd = targets_eval.reshape((conf.batch_size, conf.test_image_resize**2))
        feed_dict = model.make_inputs(data_eval_fd, targets_eval_fd)
        encode_decode = sess.run(model.y_pred, feed_dict=feed_dict)
        print("shape of predictions: {}".format(encode_decode.shape))
        # Compare original images with their reconstructions
        f, a = plt.subplots(3, conf.examples_to_show, figsize=(conf.examples_to_show, 5))
        for i in range(conf.examples_to_show):
            a[0][i].imshow(np.reshape(data_eval[i,:,:], (conf.test_image_resize, conf.test_image_resize)))
            a[1][i].imshow(np.reshape(targets_eval[i,:,:], (conf.test_image_resize, conf.test_image_resize)))
            im = a[2][i].imshow(np.reshape(encode_decode[i], (conf.test_image_resize, conf.test_image_resize)))
        plt.colorbar(im)
        plt.savefig('./autoencoder_eval_{}.png'.format(tag))

if __name__ == "__main__":
    #logging.basicConfig(filename='autoencoder.log', level=logging.DEBUG)
    mainFunc(sys.argv[1:])
