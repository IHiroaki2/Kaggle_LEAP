

Google Colabo pro+
GPU(A100)

## DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
export KAGGLE_CONFIG_DIR=[directory of kaggle.json]
cd [your LEAP directory]
# make directory & Download of kaggle data
./preparation.sh 

## make dataset (tfrecord) 
python data_splitting.py
python make_dataset_kaggle.py

## If you want to add additional data
## https://leap-stc.github.io/ClimSim/dataset.html
python download_from_hf.py
python make_dataset_hf.py

## For training
# Adjust parameters in config.py
python training.py
# Enter the path of the trained model in config.py
python inference.py
