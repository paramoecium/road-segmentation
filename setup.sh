#!/usr/bin/env bash

# Create data directories
mkdir -p data/test_set/downsampled
mkdir -p data/training/groundtruth/downsampled
mkdir -p data/training/images/downsampled

# Creates log directories for checkpoints
mkdir -p logs

# Create output directories
mkdir -p results/CNN_Output/test/raw
mkdir -p results/CNN_Output/test/high_res_raw
mkdir -p results/CNN_Output/training/raw
mkdir -p results/CNN_Output/training/high_res_raw

mkdir -p results/CNN_Output_Baseline/test/raw
mkdir -p results/CNN_Output_Baseline/training/raw

mkdir -p results/Autoencoder_Output/raw
mkdir -p results/Autoencoder_Output/high_res_raw
mkdir -p logs/Autoencoder

mkdir -p results/CNN_Autoencoder_Output/test
mkdir -p results/CNN_Autoencoder_Output/train
mkdir -p logs/CNN_Autoencoder

# Create tmp directories to hold TensorFlow results
mkdir -p tmp
mkdir -p src/baseline/tmp

# Downsample images
cp data/test_set/*.png data/test_set/downsampled
cp data/training/groundtruth/*.png data/training/groundtruth/downsampled
cp data/training/images/*.png data/training/images/downsampled
cd src
python cilutil/resizing.py
cd ..
