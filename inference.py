#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KERAS_BACKEND"] = "jax"

import datetime
import gc
import numpy as np
import pandas as pd
import polars as pl

import jax
import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf

from tensorflow.keras.utils import register_keras_serializable

from myfunctions import test_data, target_cols, preprocessing, submission

from models import Conv1D, MultiHeadsAttention, FFNN, InputConv1D, OutputTranspose, CustomModel

from config import CFG


df_test = test_data()
target_cols = target_cols()

K.clear_session()
    
norm_data = pd.read_parquet(f"data/norm_data.parquet")
mean_x = norm_data.iloc[:556, 0].values
stdd_x = norm_data.iloc[:556, 1].values
mean_y = norm_data.iloc[556:, 0].values
stdd_y = norm_data.iloc[556:, 1].values

df_test_norm = (df_test.to_numpy() - mean_x) / stdd_x

custom_objects = {"FFNN": FFNN,
                  "MultiHeadsAttention": MultiHeadsAttention,
                  "Conv1D": Conv1D,
                  "InputConv1D":InputConv1D,
                  "OutputTranspose":OutputTranspose,
                  "CustomModel":CustomModel,}

     
model = tf.keras.models.load_model(CFG.pred_model_filepath, custom_objects=custom_objects)

p_test = model.predict(df_test_norm, batch_size= CFG.batch_size)
p_test = np.array(p_test) * stdd_y + mean_y

df_p_test = preprocessing(df_test, p_test, target_cols)

submission(df_p_test, target_cols)
