#!/bin/bash
module load new gcc/4.8.2 python/3.6.0
python denoise_autoencoder.py -n 2 -t 20e
