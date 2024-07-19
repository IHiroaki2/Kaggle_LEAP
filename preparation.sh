#!/bin/bash

# pip install kaggle
# export KAGGLE_CONFIG_DIR=/content/drive/MyDrive/Kaggle
# cd /content/drive/MyDrive/Kaggle/LEAP/submit_code
# chmod +x preparation.sh


##↓ ./preparation.sh
mkdir data
mkdir tfrecord_data
mkdir nc_data
mkdir model
mkdir submission
mkdir leap-atmospheric-physics-ai-climsim

cd leap-atmospheric-physics-ai-climsim

kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f test.csv
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f sample_submission.csv
kaggle competitions download -c leap-atmospheric-physics-ai-climsim -f train.csv

cd ..