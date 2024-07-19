#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import polars as pl
import pandas as pd
import numpy as np
from myfunctions import float_feature, to_example

from config import CFG


for i in range(0, 10):
    print(i)
    
    df = np.concatenate([pd.read_parquet(f"{CFG.MAIN_FOLDER}data/data_{k}_{i}.parquet").to_numpy() for k in range(17)], axis=1) 

    tmp_dataset = tf.data.Dataset.from_tensor_slices((df))

    record_dir = "tfrecord_data/train_data_{}.tfrecord".format("{:03d}")

    shard_num = 10

    for n in range(shard_num):
        tfrecords_shard_path = os.path.join(record_dir.format(n+i*10))
        shard_data = tmp_dataset.shard(shard_num, n)

        with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
            for data in shard_data:

                data = data.numpy()
                tf_example = to_example(data[:556],data[556:])
                writer.write(tf_example)