import numpy as np
import pdb

def add_border(im, w, h, aerial_image):
    """
    Adds a padding of uninteresting pixels to the borders of an image
    so that they can be evenly broken down into pathes
    Args:
        im: image
        w: width of patch
        h: height of patch
        aerial_image: bool so decide what sort of padding to use
    Returns:
        image with borders
    """

    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3 ## 3 channels RBG
    ## creating padding in the image if there is not a whole number of patches which fit on the image
    ## creating padding for bottom of image
    rw = imgwidth % w
    pad_width = w - rw
    if rw != 0:
        if aerial_image:
            if is_2d:
                ## append means to image channels
                tmp = np.full((pad_width, imgheight), np.mean(im))
                ##pdb.set_trace()
                im_padded = np.concatenate((im,tmp), axis=0)
            else:
                ## append mean to width of image
                tmp_r = np.full((pad_width, imgheight), np.mean(im[:,:,0]))
                tmp_b = np.full((pad_width, imgheight), np.mean(im[:,:,1]))
                tmp_g = np.full((pad_width, imgheight), np.mean(im[:,:,2]))
                tmp = np.stack((tmp_r, tmp_b, tmp_g), axis=2) ## axis=2 creates new dim
                assert tmp.shape == (pad_width, imgheight, 3), 'width padding not the correct shape'
                ##pdb.set_trace()
                im_padded = np.concatenate((im, tmp), axis=0)
        else:
            ## gt images are of size (w,h) no RBG
            tmp = np.full((pad_width, imgheight), 0)
            ##pdb.set_trace()
            ##print("tmp shape: {}".format(tmp.shape))
            ##print("im shape: {}".format(im.shape))
            im_padded = np.concatenate((im,tmp), axis=0)
    else:
        im_padded = im

    ## creating padding for the right of the image
    imgwidth_new = im_padded.shape[0]
    rh = imgheight % h
    pad_height = h - rh
    if rh != 0:
        if aerial_image:
            if is_2d:
                tmp = np.full((imgwidth_new, pad_height), np.mean(im))
                im_padded = np.concatenate((im_padded,tmp), axis=1)
            else:
                ## append mean to width of image
                tmp_r = np.full((imgwidth_new, pad_height), np.mean(im[:,:,0]))
                tmp_b = np.full((imgwidth_new, pad_height), np.mean(im[:,:,1]))
                tmp_g = np.full((imgwidth_new, pad_height), np.mean(im[:,:,2]))
                tmp = np.stack((tmp_r, tmp_b, tmp_g), axis=2)
                assert tmp.shape == (imgwidth_new, pad_height, 3), 'height padding not the correct shape'
                im_padded = np.concatenate((im_padded, tmp), axis=1) ## axis = 1 to concatenate along cols
                ##Image.fromarray(img_float_to_uint8(im_padded)).save("padded.png")
                ##pdb.set_trace()
        else:
            ## gt images are of size (w,h) no RBG
            tmp = np.full((imgwidth_new, pad_height), 0)
            assert tmp.shape == (imgwidth_new, pad_height), 'height padding not the correct shape'
            im_padded = np.concatenate((im_padded,tmp), axis=1)
            ##Image.fromarray(img_float_to_uint8(im_padded)).save("padded_gt.png")
            ##pdb.set_trace()
    return im_padded

## Ref: https://github.com/mato93/road-extraction-from-aerial-images/blob/master/src/patch_extraction_module.py
def mirror_border(img, w, h):
    """ Pads an input image img with a border of size border_size using a mirror
    boundary condition
    Args:
        img: image
        w: patch width
        h: patch height
    Returns:
        img with mirror padding
    """
    assert w == h, 'the patch width and patch height need to be the same to use mirror padding function'
    rw = img.shape[0] % w
    ## padding required
    if rw != 0:
        border_size = w - rw

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
        res_img[0 : border_size, 0 : border_size] = \
            np.fliplr(np.flipud(img[0 : border_size, 0 : border_size]))
        res_img[0 : border_size, res_img.shape[1] - border_size : res_img.shape[1]] = \
            np.fliplr(np.flipud(img[0 : border_size, img.shape[1] - border_size : img.shape[1]]))
        res_img[res_img.shape[0] - border_size : res_img.shape[0], 0 : border_size] = \
            np.fliplr(np.flipud(img[img.shape[0] - border_size : img.shape[0], 0 : border_size]))
        res_img[res_img.shape[0] - border_size : res_img.shape[0], res_img.shape[1] - border_size : res_img.shape[1]] = \
            np.fliplr(np.flipud(img[img.shape[0] - border_size : img.shape[0], img.shape[1] - border_size : img.shape[1]]))

        return res_img
    ## no padding required
    else:
        return img

def img_crop(im, w, h, aerial_image):
    """
    Extracts patches from a given image
    Args:
        im: image
        w: width of patch
        h: height of patch
        aerial_image: bool so decide what sort of padding to use
    Returns:
        list with patches of images
    """

    list_patches = []

    ##im_new = add_border(im, w, h, aerial_image)
    img_new = mirror_border(im, w, h)

    imgwidth_new = img_new.shape[0]
    imgheight_new = img_new.shape[1]
    is_2d = len(im.shape) < 3 ## 3 channels RBG
    assert imgwidth_new % w == 0 and imgheight_new % h == 0, 'New img dimensions are not wholly covered by patches'
    for i in range(0,imgheight_new,h):
        for j in range(0,imgwidth_new,w):
            if is_2d:
                im_patch = img_new[j:j+w, i:i+h]
            else:
                im_patch = img_new[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

## Ref: https://github.com/mato93/road-extraction-from-aerial-images/blob/master/src/patch_extraction_module.py
## TODO: 45 deg rotations
def augment_image(img, out_ls, num_of_transformations):
    """ Augments the input image img by a number of transformations (
    rotations by 90 deg and flips).
    Args:
        img: image
        out_ls : list of output images
        num_of_transformations : number of transformations to compute
    Returns:
        out_ls with new rotated images.
    """

    if num_of_transformations > 0:
        tmp = np.rot90(img)
        out_ls.append(tmp)
    if num_of_transformations > 1:
        tmp = np.rot90(np.rot90(img))
        out_ls.append(tmp)
    if num_of_transformations > 2:
        tmp = np.rot90(np.rot90(np.rot90(img)))
        out_ls.append(tmp)

    ## Flipped rotations
    if num_of_transformations > 3:
        img2 = np.fliplr(img)
        tmp = np.rot90(img2)
        out_ls.append(tmp)
    if num_of_transformations > 4:
        tmp = np.rot90(np.rot90(img2))
        out_ls.append(tmp)
    if num_of_transformations > 5:
        tmp = np.rot90(np.rot90(np.rot90(img2)))
        out_ls.append(tmp)
    if num_of_transformations > 6:
        out_ls.append(img2)
