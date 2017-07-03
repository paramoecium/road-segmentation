#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.image as mpimg
import re

def masks_to_submission(submission_filename, *image_filenames, foreground_threshold=0.2):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for filename in image_filenames[0:]:
            img_number = int(re.search(r"\d+", filename).group(0))
            im = mpimg.imread(filename)
            patch_size = 16
            for j in range(0, im.shape[1], patch_size):
                for i in range(0, im.shape[0], patch_size):
                    patch = im[i:i + patch_size, j:j + patch_size]
                    label = 1 if np.mean(patch) > foreground_threshold else 0
                    f.writelines("{:03d}_{}_{},{}\n".format(img_number, j, i, label))

def binary_masks_to_submission(submission_filename, *image_filenames, foreground_threshold = 0.2):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for filename in image_filenames[0:]:
            img_number = int(re.search(r"\d+", filename).group(0))
            im = mpimg.imread(filename)
            patch_size = 16
            for j in range(0, im.shape[1], patch_size):
                for i in range(0, im.shape[0], patch_size):
                    patch = im[i:i + patch_size, j:j + patch_size]
                    label = 1 if np.mean(patch) > foreground_threshold else 0
                    f.writelines("{:03d}_{}_{},{}\n".format(img_number, j, i, label))
