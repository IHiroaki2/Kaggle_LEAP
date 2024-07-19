#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import numpy as np
import pandas as pd
import polars as pl
import random
import glob
import math
import os

import netCDF4

import jax
import keras

from tensorflow.keras import backend as K
import tensorflow as tf
import time

from config import CFG


#### for make datasets
def nc_to_df(mli_nc, mlo_nc):
    mli = netCDF4.Dataset(mli_nc)
    mlo = netCDF4.Dataset(mlo_nc)

    for column in CFG.TRAIN_COLS:
        if column == "state_t":
            df = np.array(mli.variables[column][:]).T
        elif column in CFG.TRAIN_COLS_60LEVEL:
            df = np.concatenate((df, np.array(mli.variables[column][:].T)), axis=1)
        else:
            df = np.concatenate((df, np.array(mli.variables[column][:].reshape((384, 1)))), axis=1)

    for column in CFG.TARGET_COLS:
        if column in CFG.TARGET_COLS_60LEVEL:
            df_ = (np.array(mlo.variables[column][:]).T - np.array(mli.variables[column][:]).T) / 1200
            df = np.concatenate((df, df_), axis=1)
        else:
            df = np.concatenate((df, np.array(mlo.variables[column][:].reshape((384, 1)))), axis=1)

    return df


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def to_example(x, targets):
    feature = {
        'x': float_feature(x),
        'targets' : float_feature(targets),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



##### For Training
def set_seed(seed=42):
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _parse_function(example_proto):
    feature_description = {
        'x': tf.io.FixedLenFeature([556], tf.float32),
        'targets': tf.io.FixedLenFeature([CFG.target_ncols], tf.float32)
    }
    e = tf.io.parse_single_example(example_proto, feature_description)
    return e['x'], e['targets']


def get_filenames(training):
    
    if training == "kaggle":
        train_files = [f"tfrecord_data/train_data_{i:003d}.tfrecord" for i in range(100)]
    
    elif training == "additional":
        train_files = [f"tfrecord_data/train_data_{i:003d}.tfrecord" for i in range(28)]
        train_files.append(f"tfrecord_data/train_data_{k*100 + i:003d}.tfrecord" for k in range(3, 9) for i in range(1, 25))
        train_files.append(f"tfrecord_data/train_data_{k*100 + i:003d}.tfrecord" for k in range(9, 10) for i in range(1, 3))
        
    elif training == "12files":
        train_files = [f"tfrecord_data/train_data_{i:003d}.tfrecord" for i in range(100)]
    
    else:
        raise ValueError


    random.shuffle(sorted(train_files))

    valid_files = [f"tfrecord_data/train_data_{i}.tfrecord" for i in ["000", "001", "002", "003", "004", "005", "006", "007"]]
    for l in valid_files:
        train_files.remove(l)

    if training == "12files":
        return train_files[:12], valid_files

    else:
        return train_files, valid_files


def get_dataset(train_files, valid_files):

    train_options = tf.data.Options()
    train_options.deterministic = True

    ds_train = (
        tf.data.Dataset.from_tensor_slices(train_files)
        .with_options(train_options)
        .shuffle(100, seed=CFG.seed)
        .interleave(
            lambda file: tf.data.TFRecordDataset(file).map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE),
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=10,
            block_length=1000,
            deterministic=True
        )
        .shuffle(4 * CFG.batch_size)
        .batch(CFG.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_valid = (
        tf.data.TFRecordDataset(valid_files)
        .map(_parse_function)
        .batch(CFG.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds_train, ds_valid


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, epochs, trained_epochs, b_steps, alpha,):
        self.initial_learning_rate = initial_learning_rate
        self.total_steps = b_steps * epochs
        self.progress = trained_epochs / epochs
        self.alpha = alpha

    def __call__(self, step):
        pi = keras.src.ops.array(math.pi, dtype="float32")
        cosine_decayed = 0.5 * (1.0 + keras.src.ops.cos(pi * (step/self.total_steps + self.progress)))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha

        return keras.src.ops.multiply(self.initial_learning_rate , decayed)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "total_steps": self.total_steps,
            "alpha": self.alpha,
            "progress": self.progress,
        }

def lr_schedule():
    return tf.keras.optimizers.schedules.CosineDecay(
    CFG.warmup_rate,
    (CFG.epochs - CFG.epochs_warmup - CFG.epochs_ending) * CFG.steps_per_epoch,
    warmup_target=CFG.learning_rate,
    warmup_steps=CFG.steps_per_epoch * CFG.epochs_warmup,
    alpha=CFG.alpha
    )



#### For inference
def test_data():
    
    df_test = (
    pl.scan_csv(f"leap-atmospheric-physics-ai-climsim/test.csv")
    .select(pl.exclude("sample_id"))
    .cast(pl.Float64)
    .collect()
    )
    
    return df_test

def target_cols():
    
    sample = pl.read_csv(f"leap-atmospheric-physics-ai-climsim/sample_submission.csv", n_rows=1)
    target_cols = sample.select(pl.exclude('sample_id')).columns
    
    return target_cols

def preprocessing(df_test, p_test, target_cols):
    
    df_p_test = pd.DataFrame(np.zeros((p_test.shape[0], 368)), columns=target_cols)
    df_p_test.iloc[:,:] = p_test
    
    ptend_q0002 = []
    for idx in range(12, 28):
        ptend_q0002.append(f"ptend_q0002_{idx}")
        df_p_test[f"ptend_q0002_{idx}"] = -df_test[f"state_q0002_{idx}"].to_numpy() / 1200
        
    
    ptend_q_list = []
    for i in range(1, 4):
        for k in range(60):
            ptend_q_list.append(f"ptend_q000{i}_{k}")
    df_test_arr = df_test.to_numpy()
    q_leverage=np.where(p_test[:,60:240]==0, 0, df_test_arr[:,60:240]/p_test[:,60:240])

    bool_arr = (q_leverage>-1200)&(q_leverage<0)
    replacement = df_test_arr[:, 60:240] / -1200
    original = df_p_test[ptend_q_list].values

    fix_origin = np.where(bool_arr, replacement, original)

    df_p_test[ptend_q_list] = fix_origin
    
    return df_p_test


def submission(df_p_test, target_cols):
    sample = pl.read_csv(f"leap-atmospheric-physics-ai-climsim/sample_submission.csv")
    submission = sample.to_pandas()
    submission[target_cols] = df_p_test.values
    pl.from_pandas(submission[["sample_id"] + target_cols]).write_csv(f"submission/submission_{CFG.trial}.csv")