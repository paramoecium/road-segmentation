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

def extract_patches(filename_base, num_images, patch_size=conf.patch_size, phase='train'):

    patches = []
    for i in range(1, num_images+1):
        if phase == 'train':
            imageid = "satImage_%.3d" % i
        if phase == 'test':
            imageid = "raw_test_%d_pixels" % i
        image_filename = filename_base + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            img = img[::16, ::16]
            patches.append(image.extract_patches(img, (patch_size, patch_size), extraction_step=1))
            if phase == 'train':
                patches.append(image.extract_patches(np.rot90(img), (patch_size, patch_size), extraction_step=1))
    return patches

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
    targets = extract_patches(train_data_filename, conf.train_size, conf.patch_size, 'train')
    targets = np.stack(targets).reshape(-1, conf.patch_size, conf.patch_size) # (n, 16, 16)
    targets = targets.reshape(len(targets), -1) # (20000, 256)
    print("Shape of targets: {}".format(targets.shape))
    validation = np.copy(targets[:conf.val_size,:])
    targets = np.copy(targets[conf.val_size:,:])

    print("Adding noise to training data")
    train = corrupt(targets, conf.corruption)
    validation = corrupt(validation, conf.corruption)

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

                batch_inputs = train[batch_indices,:]
                batch_targets = targets[batch_indices,:]
                feed_dict = model.make_inputs(batch_inputs, batch_targets)

                _, train_summary = sess.run([model.optimizer, model.summary_op], feed_dict)
                train_writer.add_summary(train_summary, global_step)

                global_step += 1
                batch_index += 1

        saver.save(sess, os.path.join(train_logfolderPath, "cnn-ae-{}-{}-ep{}-final.ckpt".format(tag_string, timestamp, conf.num_epochs)))
        print("Done with training for {} epochs".format(conf.num_epochs))

        if conf.visualise_training:
            print("Visualising encoder results and true images from train set")
            feed_dict = model.make_inputs_predict(validation)
            encode_decode = sess.run(model.y_pred, feed_dict=feed_dict) ## predictions from model are [batch_size, dim, dim, n_channels] i.e. (3125, 16, 16, 1)
            print("shape of predictions: {}".format(encode_decode.shape))
            # Compare original images with their reconstructions
            f, a = plt.subplots(2, conf.examples_to_show, figsize=(conf.examples_to_show, 5))
            for i in range(conf.examples_to_show):
                val = validation[i,:].reshape((conf.patch_size, conf.patch_size))
                pred = encode_decode[i,:,:,0]
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
            test = extract_patches(prediction_test_dir, conf.test_size, conf.patch_size, 'test')
            test = np.stack(test).reshape(-1, conf.patch_size, conf.patch_size) # (n, 16, 16)
            test = test.reshape(len(test), -1) # (n, 256)
            print("Shape of test: {}".format(test.shape)) # (26450, 256)

            # feeding in one image at a time in the convolutional autoencoder for prediction
            # where the batch size is the number of patches per test image
            predictions = []
            runs = test.shape[0] // conf.batch_size
            rem = test.shape[0] % conf.batch_size
            for i in range(runs):
                batch_inputs = test[i*conf.batch_size:((i+1)*conf.batch_size),:]
                feed_dict = model.make_inputs_predict(batch_inputs)
                prediction = sess.run(model.y_pred, feed_dict) ## numpy array (50, 76, 76, 1)
                predictions.append(prediction)
            if rems > 0:
                batch_inputs = test[runs*conf.batch_size:(runs*conf.batch_size + rems),:]
                feed_dict = model.make_inputs_predict(batch_inputs)
                prediction = sess.run(model.y_pred, feed_dict)
                predictions.append(prediction)

            # # Save outputs to disk
            # for i in range(conf.test_size):
            #     print("Test img: " + str(i+1))
            #     img_name = "cnn_ae_test_" + str(i+1)
            #     output_path = "../results/CNN_Autoencoder_Output/high_res_raw/"
            #     if not os.path.isdir(output_path):
            #         raise ValueError('no CNN data to run Convolutional Denoising Autoencoder on')
            #     print(predictions[i].shape) # (1444, 16, 16, 1)
            #     prediction = reconstruction(predictions[i][:,:,:,0], type='test')
            #     scipy.misc.imsave(output_path + img_name + ".png", prediction)
            #
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
            # plt.savefig('./cnn_autoencoder_prediction_{}.png'.format(tag))

            print("Finished saving cnn autoencoder outputs to disk")

if __name__ == "__main__":
    #logging.basicConfig(filename='autoencoder.log', level=logging.DEBUG)
    mainFunc(sys.argv[1:])
