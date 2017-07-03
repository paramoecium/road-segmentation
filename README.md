# Road Extraction from Aerial Images

# Authors

* [Florian Chlan](https://github.com/flock0)
* [Samuel Kessler](https://github.com/skezle)
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

# Folder Structure

|-- **src**

    |-- **baseline**
    
        |-- **model_baseline1.py**: defines the first baseline for the project. Provided by CIL TAs.
        
    |-- **model_baseline2.py**: defines the second baseline. Provided by previous group, cloned from [here](https://github.com/mato93/road-extraction-from-aerial-images).
    
    |-- **constants_baseline2.py**: defines the constants used for second baseline. Provided by previous group, cloned from [here](https://github.com/mato93/road-extraction-from-aerial-images).
    
    |-- **data_loading_module.py**: helper functions for baseline 2 and CNN with weighted loss. Provided by previous group, cloned from [here](https://github.com/mato93/road-extraction-from-aerial-images).
    
    |-- **patch_extraction_module.py**: helper functions for baseline 2 and CNN with weighted loss. Provided by previous group, cloned from [here](https://github.com/mato93/road-extraction-from-aerial-images).
    
    |-- **model_weightedloss.py**: defines the median frequency class balancing CNN. Adapted from [previous group](https://github.com/mato93/road-extraction-from-aerial-images)
    
    |-- **median_frequency_balancing.py**: helper function for the median frequcny class balancing CNN
    
    |-- **constants.py**: config for the median frequency class balancing CNN.
    
    |-- **autoencoder**
    
        |-- **model.py**: defines the model of the autoencoder.
        
        |-- **ae_config.py**: defines the constants for the autoencoder.
        
        |-- **denoise_autoencoder.py**: runs the autoencoder.
        
    |-- **cnn_autoencoder**
    
        |-- **model.py**: defines the convolutional autoencoder model.
        
        |-- **cnn_ae_config.py**: defines the config for the convolutional autoencoder.
        
        |-- **denoise_cnn_autoencoder.py**: runs the convolutional autoencoder.
        
    |-- **mask_to_submission.py**: Converts test images to submission csv file. Provided by CIL TAs

