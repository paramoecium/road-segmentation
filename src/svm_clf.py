import os
import time as time
import numpy as np

from skimage.transform import resize
from sklearn import svm

import patch_extraction_module as pem
import data_loading_module as dlm
import constants as const

import pdb

t = time.time()
train_data_filename = "../results/CNN_Autoencoder_Output/train/"
train_labels_filename = "../data/training/groundtruth/"

num_images = const.TRAINING_SIZE
labels = dlm.extract_label_images(train_labels_filename, num_images, const.POSTPRO_PATCH_SIZE, const.POSTPRO_PATCH_SIZE)
labels_train_cae  = dlm.read_image_array(train_data_filename, num_images, "cnn_ae_train_%.1d")

tmp = []
for i in range(len(labels_train_cae)):
    tmp.append(resize(labels_train_cae[i],
                                 (400 // const.POSTPRO_PATCH_SIZE, 400 // const.POSTPRO_PATCH_SIZE),
                                 order=0,
                                 preserve_range=True))
# extract patches and corresponding groundtruth center value
patch_size = 1
border_size = const.POSTPRO_SVM_PATCH_SIZE // 2
stride = 1
nTransforms = 0

patches_train = []
patches_labels = []
for i in range(len(labels_train_cae)):
    patches_train.extend(pem.img_crop(tmp[i], patch_size, border_size, stride, nTransforms))
    patches_labels.extend(pem.img_crop(labels[i], 1, 0, stride, nTransforms))

X = np.asarray([np.ravel(np.squeeze(np.asarray(p))) for p in patches_train])
y = np.squeeze(np.asarray(patches_labels))
print(X.shape)
print(y.shape)

# split into train and eval datasets
np.random.seed(123)
n = X.shape[0]
perm_idx = np.random.permutation(n)
split = int(0.8*n)
training_idx, test_idx = perm_idx[:split], perm_idx[split:]
training, test = X[training_idx,:], X[test_idx,:]
training_targets, test_targets = y[training_idx], y[test_idx]


elapsed = time.time() - t
print("Extracting patches from training data took: " + str(elapsed) + " s")
print("Training set size: " + str(training_targets.shape))
print("Fitting SVM...")
t = time.time()

classifier = svm.SVC()
classifier.fit(training, training_targets)

elapsed = time.time() - t
print("Training SVM took " + str(elapsed) + " s")

from sklearn.metrics import accuracy_score
y_new = classifier.predict(test)
acc = accuracy_score(test_targets, y_new)
print("Postprocessing training set accuracy: " + str(acc))

print("Prediction on Convolutional autoencoder test outputs")
test_data_filename = "../results/CNN_Autoencoder_Output/test/"
labels_test_cae  = dlm.read_image_array(test_data_filename, 50, "cnn_ae_test_%.1d")
labels_test_cae[0].shape

import scipy
output_dir = '../results/SVM/'
output_images = []
for i in range(len(labels_test_cae)):
    print()
    img = labels_test_cae[i]
    img_patches = pem.img_crop(img, 1, const.POSTPRO_SVM_PATCH_SIZE // 2, 1, 0)
    print("number of patches: {}".format(len(img_patches)))
    output = classifier.predict(np.asarray([np.ravel(np.squeeze(np.asarray(p))) for p in img_patches]))
    out_img = np.reshape(output, img.shape, order=1)
    output_images.append(out_img)
    print("Saving satImage_%d" % (i+1) + ".png")
    scipy.misc.imsave(output_dir + ("satImage_%d" % (i+1)) + ".png", resize(output_images[i], img.shape, order=0, preserve_range=True))
