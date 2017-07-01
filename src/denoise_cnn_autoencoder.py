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
from skimage.transform import resize
from sklearn.feature_extraction import image as skimg
from tqdm import tqdm
from cnn_autoencoder.model import cnn_ae, cnn_ae_ethan
from cnn_autoencoder.cnn_ae_config import Config as conf
from scipy.ndimage.interpolation import rotate

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
    elif type == 'random_neighbourhood':

        def get_neighbourhood(img, i, j):
            startPosX = i - 1
            startPosY = j - 1
            endPosX = i + 2
            endPosY = j + 2
            centerX = 1
            centerY = 1
            if i >= img.shape[0]:
                endPosX = i + 1
            if j >= img.shape[1]:
                endPosY = j + 1
            if i - 1 < 0:
                startPosX = i
                centerX = 0
            if j - 1 < 0:
                startPosY = j
                centerY = 0

            return img[startPosX:endPosX, startPosY:endPosY], (centerX, centerY)

        def randomly_flip_8_neighbourhood(data, i, j, minval, maxval, neighbour_flip_prob):
            neighbours, centerPos = get_neighbourhood(data, i, j)

            # Choose a random mask of the 8 neighbours to flip
            mask = np.random.choice([True, False], size=neighbours.shape,
                                    p=(neighbour_flip_prob, 1 - neighbour_flip_prob))
            # Certainly flip the center position
            mask[centerPos] = True

            # Depending on the center pixel we set the random neighbourhood to the min or max intensity
            replace_val = minval if neighbours[centerPos] >= 0.5 else maxval
            replace_arr = np.full(shape=neighbours.shape, fill_value=replace_val)
            neighbours[mask] = replace_arr[mask]

        RANDOMIZATIONS_NEEDED = 2
        FLIP_BASE_CHANCE = 0.5
        FLIP_8NEIGHBOURHOOD_CHANCE = 0.8
        NEIGHBOUR_FLIP_PROB_BACKGROUND = 0.025
        NEIGHBOUR_FLIP_PROB_ROAD = 0.9
        FLIP_ROAD_BACK_THRESHOLD = 0.2

        tmp = np.copy(data)
        minval = tmp.min()
        maxval = tmp.max()

        assert tmp.shape[1] == tmp.shape[2] # Assume square images
        image_width = tmp.shape[1]
        num_patches = tmp.shape[0]
        # Sample random image indices (separately for the i- and j-dimensions)
        flips_per_image= int(image_width**2 * nu)
        random_indices = np.random.randint(0, image_width, (num_patches, flips_per_image, 2))

        # Sample all the random numbers beforehand to speedup things
        flip_probabilities = np.random.random(size=(num_patches, flips_per_image, RANDOMIZATIONS_NEEDED))

        for idxpatch in range(num_patches):

            for idxcount, indices in enumerate(random_indices[idxpatch,...]):
                i, j = indices
                if flip_probabilities[idxpatch, idxcount, 0] < FLIP_BASE_CHANCE: # Apply flip only with base chance of 50 %

                    if flip_probabilities[idxpatch, idxcount, 1] < FLIP_8NEIGHBOURHOOD_CHANCE:
                        neighbour_flip_prob = NEIGHBOUR_FLIP_PROB_ROAD if tmp[idxpatch,i,j] >= FLIP_ROAD_BACK_THRESHOLD else NEIGHBOUR_FLIP_PROB_BACKGROUND
                        # Else we also flip the neighbourhood.
                        # With 50 % chance we flip either the 8- or 4-neighbourhood
                        randomly_flip_8_neighbourhood(tmp[idxpatch,...], i, j, minval, maxval, neighbour_flip_prob)
    return tmp

def load_images_to_predict(filename_base, num_images, patch_size=conf.patch_size, phase='test'):
    patches = []
    for i in range(1, num_images+1):
        if phase == 'test':
            imageid = "raw_test_%d_pixels" % i
            image_filename = filename_base + imageid + ".png"
            if os.path.isfile(image_filename):
                img = mpimg.imread(image_filename)
                img = resize(img, (38,38))
                patches.append(skimg.extract_patches(img, (patch_size, patch_size), extraction_step=1))
        elif phase == 'train_cnn_output':
            imageid = "raw_satImage_%.3d_pixels" % i
            image_filename = filename_base + imageid + ".png"
            if os.path.isfile(image_filename):
                img = mpimg.imread(image_filename)
                img = resize(img, (conf.train_image_resize,conf.train_image_resize))
                patches.append(skimg.extract_patches(img, (patch_size, patch_size), extraction_step=1))
        else:
            raise ValueError('incorrect phase')
    stacked_image_patches = np.stack(patches)
    return stacked_image_patches

def reconstruction(img_data, patches_per_predict_image_dim, size):
    """
    Reconstruct single image from flattened array.
    IMPORTANT: overlapping patches are averaged, not replaced like in recontrustion()
    Args:
        img_data: flattened image array
        type: size of the image (rescaled)
    Returns:
        recontructed image
    """

    # print("size: {}".format(size))
    # print("patches_per_dim: {}".format(patches_per_dim))
    # print("img_data: {}".format(img_data.shape))
    reconstruction = np.zeros((size,size))
    n = np.zeros((size,size))
    idx = 0
    for i in range(patches_per_predict_image_dim):
        for j in range(patches_per_predict_image_dim):
            reconstruction[i:(i+conf.patch_size),j:(j+conf.patch_size)] += img_data[idx,:,:,0]
            n[i:(i+conf.patch_size),j:(j+conf.patch_size)] += 1
            idx += 1
    return np.divide(reconstruction, n)

def binarize(image):
    """
    Binarizes an image with the threshold defined in the AE config
    :param image: The image to binarize. Most likely a low-res image where each pixel
                  represents a patch
    :return: An image where each pixel larger than the threshold is set to 1,
             and otherwise set to 0.
    """
    binarized = np.zeros(image.shape)
    binarized[image > conf.binarize_threshold] = 1
    return binarized

def resize_img(img, opt):
    """
    CNN predictions are made at the 36x36 pixel lvl and the test set needs to be at the 608x608
    lvl. The function resizes.
    Args:
        numpy array 36x36 for test or 50x50 for train
    Returns:
        numpy array 608x608 for test or 400x400 for train
    """
    #print(img.shape)
    if opt == 'test':
        size = conf.test_image_size
        blocks = conf.cnn_res # resolution of cnn output of 16x16 pixels are the same class
        steps = conf.test_image_size // blocks # 38
    elif opt == 'train':
        size = conf.train_image_size
        blocks = conf.gt_res # resolution of the gt is 8x8 pixels for one class
        steps = conf.train_image_size // blocks # 50
    else:
        raise ValueError('test or train plz')
    dd = np.zeros((size, size))
    for i in range(steps):
        for j in range(steps):
            dd[j*blocks:(j+1)*blocks,i*blocks:(i+1)*blocks] = img[j,i]
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
    train_data_directory = "../data/training/groundtruth/"

    img_indices = np.arange(conf.train_size)
    validation_img_indices = np.random.choice(conf.train_size, size=conf.val_size, replace=False)
    validation_img_mask = np.zeros(conf.train_size, dtype=bool)
    validation_img_mask[validation_img_indices] = True
    train_img_mask = np.invert(validation_img_mask)
    train_img_indices = img_indices[train_img_mask]

    uncorrupted_train_data = create_uncorrupted_data(train_data_directory, train_img_indices, conf.patch_size)
    uncorrupted_validation_data = create_uncorrupted_data(train_data_directory, validation_img_indices, conf.patch_size)

    print("Shape of training data: {}".format(uncorrupted_train_data.shape))
    patches_per_image_train = uncorrupted_train_data.shape[1] * uncorrupted_train_data.shape[2]
    print("Patches per train image: {}".format(patches_per_image_train)) # 729 for patch size 24

    uncorrupted_train_data = uncorrupted_train_data.reshape((-1, conf.patch_size, conf.patch_size))
    uncorrupted_validation_data = uncorrupted_validation_data.reshape((-1, conf.patch_size, conf.patch_size))
    print("Adding noise to training data")
    #corrupted_train_data = corrupt(uncorrupted_train_data, conf.corruption, 'random_neighbourhood')
    #corrupted_validation_data = corrupt(uncorrupted_validation_data, conf.corruption, 'random_neighbourhood')

    train = uncorrupted_train_data
    targets = uncorrupted_train_data
    print("Initializing CNN denoising autoencoder")
    # model = cnn_ae(conf.patch_size**2, ## dim of the inputs
    #                n_filters=[1, 16, 32, 64],
    #                filter_sizes=[7, 5, 3, 3],
    #                learning_rate=conf.learning_rate)
    model = cnn_ae_ethan(conf.patch_size, ## dim of the inputs Not patch_size**2
                         learning_rate=conf.learning_rate)

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
            num_batches = int(n / conf.batch_size)
            for step in tqdm(range(num_batches)):
                offset = (batch_index*conf.batch_size) % (n - conf.batch_size)
                batch_indices = perm_idx[offset:(offset + conf.batch_size)]

                batch_inputs = corrupt(train[batch_indices,:], conf.corruption, 'random_neighbourhood')
                batch_targets = targets[batch_indices,:]
                feed_dict = model.make_inputs(batch_inputs, batch_targets)

                _, train_summary = sess.run([model.optimizer, model.summary_op], feed_dict)
                train_writer.add_summary(train_summary, global_step)

                global_step += 1
                batch_index += 1

        saver.save(sess, os.path.join(train_logfolderPath, "cnn-ae-{}-{}-ep{}-final.ckpt".format(tag_string, timestamp, conf.num_epochs)))

        # Deleting train and targets objects
        del train
        del targets

        if conf.run_on_train_set:
            print("Running Convolutional Denoising Autoencoder on training images for upstream classification")
            prediction_train_dir = "../results/CNN_Output/training/high_res_raw/"
            if not os.path.isdir(prediction_train_dir):
                raise ValueError('no CNN train data to run Denoising Autoencoder on')

            print("Loading train set")
            patches_to_predict = load_images_to_predict(prediction_train_dir, conf.train_size, conf.patch_size, 'train_cnn_output')
            print("Shape of patches_to_predict: {}".format(patches_to_predict.shape))
            patches_per_predict_image_dim = patches_to_predict.shape[1] # Assume square images
            patches_to_predict = patches_to_predict.reshape((-1, conf.patch_size, conf.patch_size))
            predictions = []
            runs = patches_to_predict.shape[0] // conf.batch_size
            rem = patches_to_predict.shape[0] % conf.batch_size
            for i in tqdm(range(runs)):
                batch_inputs = patches_to_predict[i*conf.batch_size:((i+1)*conf.batch_size),...]
                feed_dict = model.make_inputs_predict(batch_inputs)
                prediction = sess.run(model.y_pred, feed_dict)
                predictions.append(prediction)
            if rem > 0:
                batch_inputs = patches_to_predict[runs*conf.batch_size:(runs*conf.batch_size + rem),...]
                feed_dict = model.make_inputs_predict(batch_inputs)
                prediction = sess.run(model.y_pred, feed_dict)
                predictions.append(prediction)

            print("individual prediction shape: {}".format(predictions[0].shape))
            predictions = np.concatenate(predictions, axis=0)
            print("Shape of predictions: {}".format(predictions.shape))

            # Save outputs to disk
            for i in range(conf.train_size):
                print("Train img: " + str(i+1))
                img_name = "cnn_ae_train_" + str(i+1)
                output_path = "../results/CNN_Autoencoder_Output/train/"
                if not os.path.isdir(output_path):
                    raise ValueError('no CNN data to run Convolutional Denoising Autoencoder on')
                prediction = reconstruction(predictions[i*patches_per_predict_image_dim**2:(i+1)*patches_per_predict_image_dim**2,:], patches_per_predict_image_dim, conf.train_image_resize)
                # resizing test images to 400x400 and saving to disk
                scipy.misc.imsave(output_path + img_name + ".png", resize_img(prediction, 'train'))

        if conf.visualise_validation:
            print("Visualising encoder results and true images from train set")
            f, a = plt.subplots(2, conf.examples_to_show, figsize=(conf.examples_to_show, 5))
            for i in range(conf.examples_to_show):
                inputs = validation[i*patches_per_image_train:(i+1)*patches_per_image_train,:]
                feed_dict = model.make_inputs_predict(inputs)
                encode_decode = sess.run(model.y_pred, feed_dict=feed_dict) ## predictions from model are [batch_size, dim, dim, n_channels] i.e. (3125, 16, 16, 1)
                print("shape of predictions: {}".format(encode_decode.shape)) # (100, 16, 16, 1)
                val = reconstruction(inputs, 50)
                pred = reconstruction(encode_decode[:,:,:,0].reshape(-1, conf.patch_size**2), 50) ## train images rescaled to 50 by 50 granularity
                a[0][i].imshow(val, cmap='gray', interpolation='none')
                a[1][i].imshow(pred, cmap='gray', interpolation='none')
                a[0][i].get_xaxis().set_visible(False)
                a[0][i].get_yaxis().set_visible(False)
                a[1][i].get_xaxis().set_visible(False)
                a[1][i].get_yaxis().set_visible(False)
            plt.gray()
            plt.savefig('./cnn_autoencoder_eval_{}.png'.format(tag))

        if conf.run_on_test_set:
            print("Running the Convolutional Denoising Autoencoder on the predictions")
            prediction_test_dir = "../results/CNN_Output/test/high_res_raw/"
            if not os.path.isdir(prediction_test_dir):
                raise ValueError('no CNN data to run Convolutional Denoising Autoencoder on')

            print("Loading test set")
            patches_per_image_test = ( (conf.test_image_size // conf.cnn_res) - conf.patch_size + 1)**2 ## 608 / 16 = 38, where 16 is the resolution of the CNN output
            print("patches per test image: {}".format(patches_per_image_test))
            test = load_images_to_predict(prediction_test_dir, conf.test_size, conf.patch_size, 'test')
            test = np.stack(test).reshape(-1, conf.patch_size, conf.patch_size) # (n, 16, 16)
            test = test.reshape(len(test), -1) # (n, 256)
            print("Shape of test: {}".format(test.shape)) # Shape of test: (26450, 256)

            predictions = []
            runs = test.shape[0] // conf.batch_size
            rem = test.shape[0] % conf.batch_size
            for i in range(runs):
                batch_inputs = test[i*conf.batch_size:((i+1)*conf.batch_size),:]
                feed_dict = model.make_inputs_predict(batch_inputs)
                prediction = sess.run(model.y_pred, feed_dict) ## numpy array (50, 76, 76, 1)
                predictions.append(prediction)
            if rem > 0:
                batch_inputs = test[runs*conf.batch_size:(runs*conf.batch_size + rem),:]
                feed_dict = model.make_inputs_predict(batch_inputs)
                prediction = sess.run(model.y_pred, feed_dict)
                predictions.append(prediction)

            print("individual prediction shape: {}".format(predictions[0].shape))
            predictions = np.concatenate(predictions, axis=0).reshape(test.shape[0], conf.patch_size**2)
            #predictions = predictions.reshape(len(predictions), -1)
            print("Shape of predictions: {}".format(predictions.shape))

            # Save outputs to disk
            for i in range(conf.test_size):
                print("Test img: " + str(i+1))
                img_name = "cnn_ae_test_" + str(i+1)
                output_path = "../results/CNN_Autoencoder_Output/test/"
                binarize_output_path = os.path.join(output_path, "binarized/")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                if not os.path.exists(binarize_output_path):
                    os.makedirs(binarize_output_path)
                prediction = reconstruction(predictions[i*patches_per_image_test:(i+1)*patches_per_image_test,:], 38) # 38 is the resized test set dim as resolution is 16x16
                binarized_prediction = binarize(prediction)
                # resizing test images to 608x608 and saving to disk
                resized_greylevel_output_images = resize_img(prediction, 'test')
                scipy.misc.imsave(output_path + img_name + ".png", resized_greylevel_output_images)

                resized_binarized_output_images = resize_img(binarized_prediction, 'test')
                scipy.misc.imsave(binarize_output_path + img_name + ".png", resized_binarized_output_images)
            f, a = plt.subplots(2, conf.examples_to_show, figsize=(conf.examples_to_show, 5))
            for i in range(conf.examples_to_show):
                t = reconstruction(test[i*patches_per_image_test:(i+1)*patches_per_image_test,:], (conf.test_image_size // conf.cnn_res)) # (conf.test_image_size // conf.cnn_res) = 38
                pred = reconstruction(predictions[i*patches_per_image_test:(i+1)*patches_per_image_test,:], 38)
                a[0][i].imshow(t, cmap='gray', interpolation='none')
                a[1][i].imshow(pred, cmap='gray', interpolation='none')
                a[0][i].get_xaxis().set_visible(False)
                a[0][i].get_yaxis().set_visible(False)
                a[1][i].get_xaxis().set_visible(False)
                a[1][i].get_yaxis().set_visible(False)
            plt.gray()
            plt.savefig('./cnn_autoencoder_prediction_{}.png'.format(tag))

            print("Finished saving cnn autoencoder test set to disk")

def create_uncorrupted_data(train_data_directory, image_indices, patch_size):
    """
    Loads the uncorrupted training (or validation) data (i.e. the groundtruth) from disk, rotates the images
    and divides them into patches with a stride of half the patch_size

    :param train_data_directory: The directory path where the groundtruth files are located
    :param image_indices: The indices of the training images to load
    :param patch_size: The desired patch sizes
    :return: A tensor of patches with dimensions
        (image_indices.size * (number of resizes + 1), vertical patch count, horizontal patch count, patch_size, patch_size)

    """
    all_image_patches = []
    for i in image_indices:
        imageid = "satImage_%.3d" % i
        image_filename = train_data_directory + imageid + ".png"

        if os.path.isfile(image_filename):
            original_img = mpimg.imread(image_filename)

            resized_img = resize(original_img, (conf.train_image_resize, conf.train_image_resize))
            rotated_images = add_rotations(resized_img)

            rotated_img_patches = [skimg.extract_patches(rotimg, (conf.patch_size, conf.patch_size), extraction_step=patch_size//2)for rotimg in rotated_images]

            all_image_patches += rotated_img_patches

    stacked_image_patches = np.stack(all_image_patches)
    return stacked_image_patches


def add_rotations(image):
    """
    Rotates the provided image a couple of time to generate more training data.
    This should make the autoencoder more robust to diagonal roads for example.

    The rotations will keep the dimensions of the image intact.

    :param image: The image to rotate
    :return: A list of rotated images, including the original image
    """
    rot90img = rotate(image, 90, reshape=False, mode='reflect', order=3)
    rot45img = rotate(image, 45, reshape=False, mode='reflect', order=3)
    rot135img = rotate(image, 135, reshape=False, mode='reflect', order=3)

    return [image, rot90img, rot45img, rot135img]

if __name__ == "__main__":
    #logging.basicConfig(filename='autoencoder.log', level=logging.DEBUG)
    mainFunc(sys.argv[1:])
