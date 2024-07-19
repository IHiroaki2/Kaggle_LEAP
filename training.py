#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KERAS_BACKEND"] = "jax"

import datetime
import gc
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
import random

import jax
import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

from myfunctions import set_seed, _parse_function, get_filenames, get_dataset, lr_schedule
from myfunctions import MyLRSchedule

from models import Conv1D, MultiHeadsAttention, FFNN, InputConv1D, OutputTranspose, CustomModel

from config import CFG


print(tf.__version__)
print(jax.__version__)


set_seed(CFG.seed)
K.clear_session()
gc.collect()

now = datetime.datetime.now(ZoneInfo("Asia/Tokyo"))
model_filepath = f"{CFG.model_folderpath}/leap_model_{CFG.trial}_{now.year}-{now.month:02d}-{now.day:02d}_{now.hour:02d}-{now.minute:02d}.keras"

if CFG.retrain:
    schedule = MyLRSchedule(CFG.learning_rate, CFG.epochs, CFG.trained_epochs, CFG.steps_per_epoch, CFG.alpha,)
    epochs = CFG.epochs - CFG.trained_epochs
else:
    schedule = lr_schedule()
    epochs = CFG.epochs

with tf.device('/GPU:0'):
# with tpu_strategy.scope():
    if CFG.retrain:
        model = tf.keras.models.load_model(CFG.trained_model_filepath, custom_objects={"FFNN": FFNN,
                                                                                        "MultiHeadsAttention": MultiHeadsAttention,
                                                                                        "Conv1D": Conv1D,
                                                                                        "InputConv1D":InputConv1D,
                                                                                        "OutputTranspose":OutputTranspose,
                                                                                        "CustomModel":CustomModel,}
                                            )
    else:
        model = CustomModel()

    optimizer = tf.keras.optimizers.Adam(schedule)
    model.compile(optimizer=optimizer, loss=CFG.loss,)

    sv = ModelCheckpoint(
        model_filepath, monitor = CFG.monitor, verbose=1, save_best_only = True,
        save_weights_only=False, mode = "min", save_freq = "epoch"
                        )

    if CFG.summary:

        # print("*" * 50)
        # model.summary()
        print("*" * 50)
        if CFG.retrain:
            plt.plot([schedule(it) for it in range(0, (CFG.epochs - CFG.trained_epochs) * CFG.steps_per_epoch, CFG.steps_per_epoch)])
            plt.show()
        else:
            plt.plot([schedule(it) for it in range(0, CFG.epochs * CFG.steps_per_epoch, CFG.steps_per_epoch)])
            plt.show()
        print("*" * 50)

    train_files, valid_files = get_filenames(CFG.training)
    ds_train, ds_valid = get_dataset(train_files, valid_files)
    history = model.fit(ds_train, validation_data = ds_valid, epochs=epochs, verbose=1, callbacks = [sv],)
