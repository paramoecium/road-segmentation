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
from model import cnn_ae_model
from cnn_ae_config import Config as conf
from scipy.ndimage.interpolation import rotate
import shutil

# Random seed for reproducibility. May be removed
tf.set_random_seed(123)
np.random.seed(123)

def corrupt(data, nu, type='salt_and_pepper'):
    """
    Corrupts the data that serves as training data for the de-noising convolutinoal autoencoder

    Args:
        data: numpy array of size (num_patches, img_size, img_size)
        nu: corruption level
        type: Type of noise that should be generated.
              Currently 'salt_and_pepper' and 'random_neighbourhood' are supported
    Returns:
        numpy array of size (num_patches, img_size, img_size)
    """

    if type == 'salt_and_pepper':
        # Apply typical salt and pepper noise, flipping a random subset of the pixels
        img_max = np.ones(data.shape, dtype=bool)
        tmp = np.copy(data)
        img_max[data <= 0.5] = False
        img_min = np.logical_not(img_max)
        idx = np.random.choice(a = [True, False], size=data.shape, p=[nu, 1-nu])
        tmp[np.logical_and(img_max, idx)] = 0
        tmp[np.logical_and(img_min, idx)] = 1
    elif type == 'random_neighbourhood':

        def get_neighbourhood(img, i, j):
            """
            Auxiliary function. Given the input img, outputs the 3x3 neighbourhood of the pixel indexed by i,j.
            If i or j lie on the border of the img, then the neighbourhood might be smaller than 3x3. (no padding is used)

            :param img: The input image to get the 3x3 neighbourhood from
            :param i: The first-dimension index of the center pixel
            :param j: The second-dimension index of the center pixel
            :return: An array of max 3x3 size of the neighbouring pixels, A tuple setting the center pixel in the returned neighbourhood array
            """

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
            """
            Looks at the 8-neighbourhood and randomly flips the neighbours.

            :param data: The original image
            :param i: The first-dimension index of the center pixel for which to look at the neighbourhood
            :param j: The second-dimension index of the center pixel for which to look at the neighbourhood
            :param minval: The minimum intensity level in the original image.
                           Flipped high-intensity pixels will be set to this value.
            :param maxval: The maximum intensity level in the original image.
                           Flipped low-intensity pixels will be set to this value.
            :param neighbour_flip_prob: The probability with which each of the neighbouring pixels will be flipped.
            """

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

        FLIP_8NEIGHBOURHOOD_CHANCE = 0.4 # Chance that the randomly selected pixel will be flipped.
        NEIGHBOUR_FLIP_PROB_BACKGROUND = 0.025 # Given that the center pixel is a background pixel, with what probability do we flip the neighbours?
        NEIGHBOUR_FLIP_PROB_ROAD = 0.9 # Given that the center pixel is a road pixel, with what probability do we flip the neighbours?
        FLIP_ROAD_BACK_THRESHOLD = 0.2 # Threshold for a pixel to be considered a road pixel for corruption.

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
        flip_probabilities = np.random.random(size=(num_patches, flips_per_image))

        for idxpatch in range(num_patches):
            for idxcount, indices in enumerate(random_indices[idxpatch,...]):
                i, j = indices

                # From the randomly selected pixels, only a random subset will be considered for flipping
                if flip_probabilities[idxpatch, idxcount] < FLIP_8NEIGHBOURHOOD_CHANCE:
                    # Consider differing flipping probabilities for road or background pixels
                    neighbour_flip_prob = NEIGHBOUR_FLIP_PROB_ROAD if tmp[idxpatch,i,j] >= FLIP_ROAD_BACK_THRESHOLD \
                                                                   else NEIGHBOUR_FLIP_PROB_BACKGROUND

                    randomly_flip_8_neighbourhood(tmp[idxpatch,...], i, j, minval, maxval, neighbour_flip_prob)
    else:
        raise ValueError('Unsupported noise type')
    return tmp

def load_patches_to_predict(directory_path, num_images, patch_size=conf.patch_size, phase='test'):
    """
    Loads prediction images and splits them up into patches.

    :param directory_path: The directory to load images from
    :param num_images: Number of images to load
    :param patch_size: The desired patch size. For prediction, the stride will be 1.
    :param phase: Whether the image to load are from the test or training dataset.
                  Must be 'test' or 'train_cnn_output'.
                  (This is important for the filename and resizing size.)

    :return: A tensor of patches with dimensions
        (num_images, vertical patch count, horizontal patch count, patch_size, patch_size)
    """
    patches = []
    if phase == 'test':
        base_filename = "raw_test_%d_pixels"
        resize_size = conf.test_image_resize
    elif phase == 'train_cnn_output':
        base_filename = "raw_satImage_%.3d_pixels"
        resize_size = conf.train_image_resize
    else:
        raise ValueError('Unsupported phase')

    for i in range(1, num_images+1):
        imageid = base_filename % i
        image_filename = directory_path + imageid + ".png"

        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            # Resize images s.t. one patch is represented by a single pixel
            img = resize(img, (resize_size, resize_size))

            # For prediction we always extract patches with stride 1 and then average the predictions
            patches.append(skimg.extract_patches(img, (patch_size, patch_size), extraction_step=1))
    stacked_image_patches = np.stack(patches)
    return stacked_image_patches

def reconstruct_image_from_patches(img_data, patches_per_predict_image_dim, size):
    """
    Reconstruct single image from multiple image patches.
    IMPORTANT: overlapping patches are averaged

    Args:
        img_data: An array with dimensions (patches_per_predict_image_dim**2, patch size, patch size)
        patches_per_predict_image_dim: Number of patches for one dimension. We assume image have the same
                                       dimension horizontally as well as vertically.
        size: Height/Widgth of the target image.
    Returns:
        recontructed image: An image of (size x size) reconstructed from the patches
    """

    reconstruction = np.zeros((size,size))
    n = np.zeros((size,size))
    idx = 0

    # Loop through all the patches in 2-dim and sum up the pixel values.
    # (We split up the image with stride 1 before)
    # Also keep a count array
    for i in range(patches_per_predict_image_dim):
        for j in range(patches_per_predict_image_dim):
            reconstruction[i:(i+conf.patch_size),j:(j+conf.patch_size)] += img_data[idx,:,:,0]
            n[i:(i+conf.patch_size),j:(j+conf.patch_size)] += 1
            idx += 1

    #Return the arithmetic average
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

def resize_img(img, mode):
    """
    CNN predictions are made at the 38x38 pixel lvl and the test set needs to be at the 608x608
    lvl. The function resizes.
    Args:
        numpy array 38x38for test or 25x25 for train
    Returns:
        numpy array 608x608 for test or 400x400 for train
    """

    if mode == 'test':
        size = conf.test_image_size
        blocks = conf.cnn_res # resolution of cnn output of 16x16 pixels are the same class
        steps = conf.test_image_size // blocks # 38
    elif mode == 'train':
        size = conf.train_image_size
        blocks = conf.gt_res # resolution of the gt is 16x16 pixels for one class
        steps = conf.train_image_size // blocks # 25
    else:
        raise ValueError("Invalid mode. Only 'train' or 'test' are supported.")
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

    # Setting the number of CPU cores is only relevant for Euler.
    print("Executing autoencoder with {} CPU cores".format(num_cores))
    if num_cores != -1:
        # We set the op_parallelism_threads in the ConfigProto and pass it to the TensorFlow session
        configProto = tf.ConfigProto(inter_op_parallelism_threads=num_cores,
                                     intra_op_parallelism_threads=num_cores)
    else:
        configProto = tf.ConfigProto()

    print("loading ground truth data")
    train_data_directory = "../data/training/groundtruth/"

    # We randomly choose the validation images
    img_indices = np.arange(conf.train_size)
    validation_img_indices = np.random.choice(conf.train_size, size=conf.val_size, replace=False)
    validation_img_mask = np.zeros(conf.train_size, dtype=bool)
    validation_img_mask[validation_img_indices] = True
    train_img_mask = np.invert(validation_img_mask)
    train_img_indices = img_indices[train_img_mask]

    # Create the uncorrupted images from the groundtruth
    uncorrupted_train_data = create_uncorrupted_data(train_data_directory, train_img_indices, conf.patch_size)
    uncorrupted_validation_data = create_uncorrupted_data(train_data_directory, validation_img_indices, conf.patch_size)

    print("Shape of training data: {}".format(uncorrupted_train_data.shape))
    patches_per_image_train = uncorrupted_train_data.shape[1] * uncorrupted_train_data.shape[2]
    print("Patches per train image: {}".format(patches_per_image_train))

    # Reshape to get rid of unnecessary dimensions
    uncorrupted_train_data = uncorrupted_train_data.reshape((-1, conf.patch_size, conf.patch_size))
    uncorrupted_validation_data = uncorrupted_validation_data.reshape((-1, conf.patch_size, conf.patch_size))

    train = uncorrupted_train_data
    targets = uncorrupted_train_data
    validation = uncorrupted_validation_data

    print("Initializing CNN denoising autoencoder")
    model = cnn_ae_model(conf.patch_size, learning_rate=conf.learning_rate)

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
        for i in tqdm(range(conf.num_epochs)):
            n = train.shape[0]
            perm_idx = np.random.permutation(n)
            batch_index = 1
            for step in range(int(n / conf.batch_size)):
                offset = (batch_index*conf.batch_size) % (n - conf.batch_size)
                batch_indices = perm_idx[offset:(offset + conf.batch_size)]

                # Corrupt the CNN input, but keep the training targets intact
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
            predict_on_train_set(model, sess)

        if conf.run_on_test_set:
            predict_on_test_set(model, sess)


            print("Finished saving cnn autoencoder test set to disk")

def predict_on_train_set(model, sess):
    print("Running Convolutional Denoising Autoencoder on training images for upstream classification")
    prediction_train_dir = "../results/CNN_Output/training/high_res_raw/"
    if not os.path.exists(prediction_train_dir):
        raise ValueError("Couldn't find directory {}".format(prediction_train_dir))

    print("Loading train set")
    patches_to_predict = load_patches_to_predict(prediction_train_dir, conf.train_size, conf.patch_size,
                                                 'train_cnn_output')
    print("Shape of training patches_to_predict: {}".format(patches_to_predict.shape))
    patches_per_predict_image_dim = patches_to_predict.shape[1]  # Assume square images
    patches_to_predict = patches_to_predict.reshape((-1, conf.patch_size, conf.patch_size))
    predictions = []
    runs = patches_to_predict.shape[0] // conf.batch_size
    rem = patches_to_predict.shape[0] % conf.batch_size
    for i in tqdm(range(runs)):
        batch_inputs = patches_to_predict[i * conf.batch_size:((i + 1) * conf.batch_size), ...]
        feed_dict = model.make_inputs_predict(batch_inputs)
        prediction = sess.run(model.y_pred, feed_dict)
        predictions.append(prediction)
    if rem > 0:
        batch_inputs = patches_to_predict[runs * conf.batch_size:(runs * conf.batch_size + rem), ...]
        feed_dict = model.make_inputs_predict(batch_inputs)
        prediction = sess.run(model.y_pred, feed_dict)
        predictions.append(prediction)

    print("individual training prediction shape: {}".format(predictions[0].shape))
    predictions = np.concatenate(predictions, axis=0)
    print("Shape of predictions on training set: {}".format(predictions.shape))

    output_path = "../results/CNN_Autoencoder_Output/train/"

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Save outputs to disk
    for i in range(conf.train_size):
        print("Train img: " + str(i + 1))
        img_name = "cnn_ae_train_" + str(i + 1)

        prediction = reconstruct_image_from_patches(
            predictions[i * patches_per_predict_image_dim ** 2:(i + 1) * patches_per_predict_image_dim ** 2, :],
            patches_per_predict_image_dim, conf.train_image_resize)
        # resizing test images to 400x400 and saving to disk
        scipy.misc.imsave(output_path + img_name + ".png", resize_img(prediction, 'train'))

def predict_on_test_set(model, sess):
    print("Running the Convolutional Denoising Autoencoder on the predictions")
    prediction_test_dir = "../results/CNN_Output/test/high_res_raw/"
    if not os.path.isdir(prediction_test_dir):
        raise ValueError("Couldn't find directory {}".format(prediction_test_dir))

    patches_to_predict = load_patches_to_predict(prediction_test_dir, conf.train_size, conf.patch_size, 'test')
    print("Shape of patches_to_predict for training data: {}".format(patches_to_predict.shape))
    patches_per_predict_image_dim = patches_to_predict.shape[1]  # Assume square images
    patches_to_predict = patches_to_predict.reshape((-1, conf.patch_size, conf.patch_size))
    predictions = []
    runs = patches_to_predict.shape[0] // conf.batch_size
    rem = patches_to_predict.shape[0] % conf.batch_size
    for i in tqdm(range(runs)):
        batch_inputs = patches_to_predict[i * conf.batch_size:((i + 1) * conf.batch_size), ...]
        feed_dict = model.make_inputs_predict(batch_inputs)
        prediction = sess.run(model.y_pred, feed_dict)
        predictions.append(prediction)
    if rem > 0:
        batch_inputs = patches_to_predict[runs * conf.batch_size:(runs * conf.batch_size + rem), ...]
        feed_dict = model.make_inputs_predict(batch_inputs)
        prediction = sess.run(model.y_pred, feed_dict)
        predictions.append(prediction)

    print("individual training image prediction shape: {}".format(predictions[0].shape))
    predictions = np.concatenate(predictions, axis=0)
    print("Shape of training image predictions: {}".format(predictions.shape))

    output_path = "../results/CNN_Autoencoder_Output/test/"
    binarize_output_path = os.path.join(output_path, "binarized/")
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    os.makedirs(binarize_output_path)

    # Save outputs to disk
    for i in range(conf.test_size):
        print("Test img: " + str(i + 1))
        img_name = "cnn_ae_test_" + str(i + 1)

        prediction = reconstruct_image_from_patches(
            predictions[i * patches_per_predict_image_dim ** 2:(i + 1) * patches_per_predict_image_dim ** 2, :],
            patches_per_predict_image_dim, conf.test_image_resize)
        binarized_prediction = binarize(prediction)
        # resizing test images to 608x608 and saving to disk
        resized_greylevel_output_images = resize_img(prediction, 'test')
        scipy.misc.imsave(output_path + img_name + ".png", resized_greylevel_output_images)

        resized_binarized_output_images = resize_img(binarized_prediction, 'test')
        scipy.misc.imsave(binarize_output_path + img_name + ".png", resized_binarized_output_images)
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

            rotated_img_patches = [skimg.extract_patches(rotimg, (conf.patch_size, conf.patch_size), extraction_step=1)for rotimg in rotated_images]

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
