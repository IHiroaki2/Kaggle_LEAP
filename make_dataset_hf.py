#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from huggingface_hub import hf_hub_download
import netCDF4
import tensorflow as tf
import gc

import os
import polars as pl
import pandas as pd
import numpy as np

from myfunctions as nc_to_df, float_feature, to_example 

from config import CFG


for year in ["3", "4", "5", "6", "7", "8", "9"]:
    if year == "9":
        m = ["01"]
    else:
        m = ["01", "02", "03", "04" ,"05", "06", "07", "08", "09", "10", "11", "12"]

    for month in m:
        days = 31
        if month in ["01", "03", "05", "07", "08", "10", "12"]:
            days = 31
        elif month in ["04", "06", "09", "11"]:
            days = 30
        else:
            days = 28

        print(f"00{year}_{month}")
        data = np.concatenate([nc_to_df(
                            f"nc_data/train/000{year}-{month}/E3SM-MMF.mli.000{year}-{month}-{day:02d}-{time*3600:05d}.nc",
                            f"nc_data/train/000{year}-{month}/E3SM-MMF.mlo.000{year}-{month}-{day:02d}-{time*3600:05d}.nc")
                            for day in range(1, days+1) for time in range(24)])


        norm_data = pd.read_parquet(f"data/norm_data.parquet")
        sub = pl.read_csv(f"leap-atmospheric-physics-ai-climsim/sample_submission.csv", n_rows=1).to_numpy()[0, 1:]
        data[:, 556:] = data[:, 556:]*sub.reshape((1, -1))

        mean = norm_data.iloc[:, 0].values
        stdd = norm_data.iloc[:, 1].values

        data = (data - mean) / stdd
        print(np.sum(np.isnan(data)))
        data = np.nan_to_num(data)
        print(np.sum(np.isnan(data)))

        data = data.astype("float32")

        tmp_dataset = tf.data.Dataset.from_tensor_slices((data))

        shard_num = 2

        for n in range(shard_num):
            num += 1
            tfrecords_shard_path = os.path.join(f"tfrecord_data/train_data_{year}{num:02d}.tfrecord")
            shard_data = tmp_dataset.shard(shard_num, n)

            with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
                for data in shard_data:

                    data = data.numpy()
                    tf_example = to_example(data[:556],data[556:])
                    writer.write(tf_example)
