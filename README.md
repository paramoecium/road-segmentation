# Road Extraction from Aerial Images

##  Autoencoder

Post post-processing of the CNN outputs is performed by a simple denoising autoencoder. The groundtruth is corrupted and and passed as input and the original groundtruth is the network's target. As a result the autoencoder is able to learn a mapping from noisy groundtruth images to clean ground truth images.
