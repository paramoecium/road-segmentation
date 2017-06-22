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


def corrupt(data, nu, type='masking_noise'):
    """
    Corrupts the data for inputing into the de-noising autoencoder

    Args:
        data: numpy array of size (num_points, 1, img_size, img_size)
        nu: corruption level
    Returns:
        numpy array of size (num_points, 1, img_size, img_size)
    """
    if type == 'masking_noise':
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
                               num_images=conf.train_size, ## TODO: change to 100 for full run
                               num_of_transformations=0,
                               patch_size=conf.image_size,
                               patch_stride=conf.image_size,
                               border_size=0,
                               zero_center=False)

    print("Resizing ground truth images so that patches from CNN are now pixels")
    targets_patch_lvl = np.zeros((targets.shape[0], targets.shape[2] // conf.post_process_patch_size, targets.shape[3] // conf.post_process_patch_size))
    for i in range(targets.shape[0]):
        targets_patch_lvl[i,:,:] = resize(targets[i,0,:,:],
                                          (targets[i,0,:,:].shape[0] // conf.post_process_patch_size, targets[i,0,:,:].shape[1] // conf.post_process_patch_size),
                                          order=0, preserve_range=True)

    print("New shape of each image: {}".format(targets_patch_lvl.shape)) ## (5, 50, 50)

    print("Deleting original data to free space")
    del targets

    # print("corrupting the ground truth labels")
    # train = corrupt(targets_patch_lvl, 0.05)
    #
    # f, a = plt.subplots(nrows=2, ncols=4, figsize=(4, 4))
    # for i in range(4):
    #     a[0][i].imshow(np.reshape(targets[i,:,:,:], (targets.shape[2], targets.shape[3])), vmin=0, vmax=1)
    #     im = a[1][i].imshow(np.reshape(train[i,:,:], (train.shape[1], train.shape[2])), vmin=0, vmax=1)
    # plt.colorbar(im)
    # plt.savefig('./ae_patching.png')

    print("Initializing model")
    print("Input size: {}".format(int(conf.train_image_size*conf.train_image_size)))
    print("H1 size: {}".format(int(conf.train_image_size*conf.train_image_size/2)))
    print("H2 size: {}".format(int(conf.train_image_size*conf.train_image_size/2/2)))
    print("H3 size: {}".format(int(conf.train_image_size*conf.train_image_size/2/2/2)))
    print("H4 size: {}".format(int(conf.train_image_size*conf.train_image_size/2/2/2/2)))
    model = ae(n_input=int(conf.train_image_size*conf.train_image_size),
               n_hidden_1=int(conf.train_image_size*conf.train_image_size/2),
               n_hidden_2=int(conf.train_image_size*conf.train_image_size/2/2),
               n_hidden_3=int(conf.train_image_size*conf.train_image_size/2/2/2),
               n_hidden_4=int(conf.train_image_size*conf.train_image_size/2/2/2/2),
               learning_rate=1e-3,
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
            print("Training epoch {}".format(i))
            logging.info("Training epoch {}".format(i))
            print("Time elapsed:    %.3fs" % (time.time() - start))
            logging.info("Time elapsed:    %.3fs" % (time.time() - start))

            print("corrupting the ground truth labels")
            train = corrupt(targets_patch_lvl,
                            float(np.random.choice(a = [0.01, 0.05], size=1, p=[0.75, 0.25])))

            perm_idx = np.random.permutation(conf.train_size)
            batch_index = 1
            for step in range(int(conf.train_size / conf.batch_size)):
                offset = (batch_index*conf.batch_size) % (conf.train_size - conf.batch_size)
                batch_indices = perm_idx[offset:(offset + conf.batch_size)]

                batch_inputs = train[batch_indices,:,:].reshape((conf.batch_size, conf.train_image_size**2))
                batch_targets = targets_patch_lvl[batch_indices,:,:].reshape((conf.batch_size, conf.train_image_size**2))

                print("shape of batch inputs: {0} and outputs: {1}".format(batch_inputs.shape, batch_targets.shape))

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
        d = train[batch_indices,:,:].reshape((conf.batch_size, conf.train_image_size**2))
        t = targets_patch_lvl[batch_indices,:,:].reshape((conf.batch_size, conf.train_image_size**2))
        feed_dict = model.make_inputs(d, t)
        encode_decode = sess.run(model.y_pred, feed_dict=feed_dict)
        print("shape of predictions: {}".format(encode_decode.shape))
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, conf.examples_to_show, figsize=(conf.examples_to_show, 4))
        for i in range(conf.examples_to_show):
            a[0][i].imshow(np.reshape(targets_patch_lvl[i,:,:], (conf.train_image_size, conf.train_image_size)))
            im = a[1][i].imshow(np.reshape(encode_decode[i].reshape(conf.train_image_size, conf.train_image_size), (conf.train_image_size, conf.train_image_size))) ## order - 'F'?
        plt.colorbar(im)
        plt.savefig('./autoencoder_eval.png')

if __name__ == "__main__":
    logging.basicConfig(filename='autoencoder.log', level=logging.DEBUG)
    mainFunc(sys.argv[1:])
