import matplotlib.image as mpimg
from skimage.transform import resize
from autoencoder.ae_config import Config as conf
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Upscale from image where patches are wxh to imgwidth x imgheight image
    """
    array_labels = np.zeros((imgwidth, imgheight))
    ii = 0
    for i in range(0, imgheight, h):
        jj = 0
        for j in range(0, imgwidth, w):
            array_labels[j:j + w, i:i + h] = labels[jj,ii]
            jj += 1
        ii += 1
    return array_labels

def img_to_label(imgwidth, imgheight, w, h, labels):
    """
    Downscale from image where patches are of wxh are single value
    Args:
        imgweight: downscaled imgwidth
        imgweight: downscaled imgheight
    """
    assert imgwidth < labels.shape[0]
    assert imgheight < labels.shape[1]
    array_labels = np.zeros((imgwidth, imgheight), dtype='int32')
    ii = 0
    for i in range(0, labels.shape[1], h):
        jj = 0
        for j in range(0, labels.shape[0], w):
            array_labels[jj, ii] = labels[j,i]
            jj += 1
        ii += 1
    return array_labels

def image2binary(img):
    img_max = np.ones(img.shape, dtype=bool)
    tmp = np.copy(img)
    tmp = np.array(tmp, dtype='int32')
    img_max[img < 1] = False
    img_min = np.logical_not(img_max)
    tmp[img_max] = 1
    tmp[img_min] = 0
    return tmp


if __name__ == "__main__":
    print("Reading random image")
    img = mpimg.imread('../results/CNN_Output/test/raw/raw_test_1_patches.png')
    print("Size of image: {}".format(img.shape))
    print(img)
    print(np.unique(img))

    plt.imshow(img)
    plt.savefig("./img.png")

    img = image2binary(img)
    plt.imshow(img)
    plt.savefig("./img2bin.png")

    print("Downscaling image")
    downscaling = img_to_label(76, 76, 8, 8, img)
    print(downscaling)
    print(np.unique(downscaling))

    im3 = plt.imshow(downscaling)
    plt.savefig("./downscale.png")
    plt.colorbar(im3)

    tmp = resize(img,(76, 76), order=0, preserve_range=True)
    im = plt.imshow(tmp)
    plt.savefig("./scikit-resize.png")
    print("tmp shape: {}".format(tmp.shape))

    # Show per pixel probabilities
    upscaling = label_to_img(608, 608, 8, 8, downscaling)
    print(upscaling.shape)

    im2 = plt.imshow(upscaling)
    plt.savefig("./upscale.png")

    print("Upscaling and original img the same: {}".format(np.array_equal(img, upscaling)))
