# Road Extraction from Aerial Images

# Authors

* [Florian Chlan](https://github.com/flock0)
* [amuel Kessler](https://github.com/skezle)
* [Yu-chen Tsai](https://github.com/paramoecium)

# Acknowledgements

This repo was cloned from https://github.com/mato93/road-extraction-from-aerial-images.

# Dependencies

* Python 3.5.3+
* Tensorflow 1.1.0
* Numpy
* scikit-learn
* scikit-image
* Pillow
* Matplotlib
* Tqdm

# How to run

1. bash setup.sh
2. cd src
3. python3 run.py

# Results

The result of running the above will create two submission csv files. submission_cae_patchsize24.csv includes the final results of the median frequency class balancing (MFCB) CNN and the denoising convolutional autoencoder, and the submission_no_postprocessing.csv contains the predictions of the MFCB CNN only.

