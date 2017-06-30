import os
import time as time
import numpy as np

import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.transform import rotate
from sklearn.feature_extraction import image
from sklearn import svm
import constants as const


def mirror_border(img, border_size):
    """ Pads an input image img with a border of size border_size using a mirror boundary condition """

    if len(img.shape) < 3:
        # Binary image
        res_img = np.zeros((img.shape[0] + 2 * border_size, img.shape[1] + 2 * border_size))
    else:
        # 3 channel image
        res_img = np.zeros((img.shape[0] + 2 * border_size, img.shape[1] + 2 * border_size, 3))
    for i in range(border_size):
        res_img[border_size : res_img.shape[0] - border_size, border_size - 1 - i] = img[:, i]                                     # left columns
        res_img[border_size : res_img.shape[0] - border_size, res_img.shape[1] - border_size + i] = img[:, img.shape[1] - 1 - i]   # right columns
        res_img[border_size - 1 - i, border_size : res_img.shape[1] - border_size] = img[i, :]                                     # top rows
        res_img[res_img.shape[0] - border_size + i, border_size : res_img.shape[1] - border_size] = img[img.shape[0] - 1 - i, :]   # bottom rows
    res_img[border_size : res_img.shape[0] - border_size, border_size : res_img.shape[1] - border_size] = np.copy(img)
    # Corners
    res_img[0 : border_size, 0 : border_size] =         np.fliplr(np.flipud(img[0 : border_size, 0 : border_size]))
    res_img[0 : border_size, res_img.shape[1] - border_size : res_img.shape[1]] =         np.fliplr(np.flipud(img[0 : border_size, img.shape[1] - border_size : img.shape[1]]))
    res_img[res_img.shape[0] - border_size : res_img.shape[0], 0 : border_size] =         np.fliplr(np.flipud(img[img.shape[0] - border_size : img.shape[0], 0 : border_size]))
    res_img[res_img.shape[0] - border_size : res_img.shape[0], res_img.shape[1] - border_size : res_img.shape[1]] =         np.fliplr(np.flipud(img[img.shape[0] - border_size : img.shape[0], img.shape[1] - border_size : img.shape[1]]))

    return res_img

def _extract_labels(num_images):
    labels = []
    train_labels_filename = "../data/training/groundtruth/"
    for i in range(1,num_images+1):
        image_file_name = train_labels_filename + "satImage_" + '%.3d' % i + ".png"
        img = mpimg.imread(image_file_name)
        img = resize(img, (50,50))
        labels.append(image.extract_patches(img, (1, 1), extraction_step=1))
        labels.append(image.extract_patches(np.rot90(img), (1, 1), extraction_step=1))
        labels.append(image.extract_patches(rotate(img, 45, resize=False, mode='reflect'),
                                              (1, 1), extraction_step=1))
    return labels

def _extract_data_svm(filename_base, num_images,
                 patch_stride=1, border_size=6,
                 phase='train'):
    """Extract patches from images."""
    patches_x = []
    for i in range(1, num_images+1):
        if phase == 'test':
            imageid = "cnn_ae_test_" + str(i)
        elif phase == 'train':
            imageid = 'cnn_ae_train_' + str(i)
        else:
            raise ValueError('test or train phase plz')
        image_filename = filename_base + imageid + ".png"
        if not os.path.isfile(image_filename):
            raise ValueError('no matching filename')
        else:
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            if phase == 'test':
                img = resize(img, (38,38))
                imgb = mirror_border(img, border_size)
                patch_size = 2*border_size + 1
                patches_x.append(image.extract_patches(imgb, (patch_size, patch_size), extraction_step=1))

            elif phase == 'train':
                img = resize(img, (50,50))
                ## patch lvl extraction
                imgb = mirror_border(img, border_size)
                patch_size = 2*border_size + 1
                patches_x.append(image.extract_patches(imgb, (patch_size, patch_size), extraction_step=1))
                patches_x.append(image.extract_patches(np.rot90(imgb), (patch_size, patch_size), extraction_step=1))
                patches_x.append(image.extract_patches( rotate(imgb, 45, resize=False, mode='reflect'),
                                                      (patch_size, patch_size), extraction_step=1))
    return patches_x


t = time.time()
train_data_filename = "../results/CNN_Autoencoder_Output/train/"
train_labels_filename = "../data/training/groundtruth/"

# ground truth label images and CNN output
dd = _extract_data_svm(train_data_filename, num_images=100,
                          patch_stride=1,
                          border_size=5,
                          phase='train')

d = np.stack(dd).reshape([-1,11*11])
print(d.shape)

tt = _extract_labels(100)
t = np.stack(tt).reshape([-1])
print(t.shape)


# In[10]:


t_bin = np.zeros(t.shape)
t_bin[t > 0.25] = 1
print(np.unique(t_bin))


# In[11]:


# split into train and eval datasets
np.random.seed(123)
n = d.shape[0]
perm_idx = np.random.permutation(n)
split = int(0.8*n)
training_idx, test_idx = perm_idx[:split], perm_idx[split:]
training, test = d[training_idx,:], d[test_idx,:]
training_targets, test_targets = t_bin[training_idx], t_bin[test_idx]
print(training.shape)
print(training_targets.shape)

print("Fitting SVM...")
t = time.time()

clf = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
clf.fit(training, training_targets)

elapsed = time.time() - t
print("Training SVM took " + str(elapsed) + " s")

from sklearn.metrics import f1_score
y_new = clf.predict(test)
acc = f1_score(test_targets, y_new)
print("Postprocessing training set f1 score: " + str(acc))

if False:
    print("Prediction on Convolutional autoencoder test outputs")
    test_data_filename = "../results/CNN_Autoencoder_Output/test/"
    dd = _extract_data_svm(test_data_filename, num_images=2,
                          patch_stride=1,
                          border_size=5,
                          phase='test')
    d = np.stack(dd).reshape([-1,11*11])
    import scipy
    output_dir = '../results/SVM/'
    output_images = []
    num_preds_per_image = 1444 # 38**2
    for i in range(2):
        img = d[i*num_preds_per_image:(i+1)*num_preds_per_image, :]
        output = clf.predict(img)
        print("shape of output: {}".format(output.shape))
        out_img = np.reshape(output, (38,38))
        print("Saving satImage_%d" % (i+1) + ".png")
        scipy.misc.imsave(output_dir + ("satImage_%d" % (i+1)) + ".png", resize(out_img, (608,608), order=0, preserve_range=True))
