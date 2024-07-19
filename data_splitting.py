#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gc
import time

import os
import polars as pl
import pandas as pd
import numpy as np

from config import CFG

columns = pl.read_csv(f"{CFG.MAIN_FOLDER}leap-atmospheric-physics-ai-climsim/train.csv", n_rows = 1).to_pandas().columns.to_list()

norm_data = pd.DataFrame(np.zeros((556+368, 2)), columns=["mean", "std"])

for i in range(0, len(CFG.start_num)):
    print(i)
    cols = columns[CFG.start_num[i] : CFG.end_num[i]]
    df = pl.read_csv(f"{CFG.MAIN_FOLDER}leap-atmospheric-physics-ai-climsim/train.csv", columns = cols).to_pandas()
    if i >= 10:
        sub = pl.read_csv(f"{CFG.MAIN_FOLDER}leap-atmospheric-physics-ai-climsim/sample_submission.csv", columns=cols, n_rows=1).to_numpy()
        df = df*sub.reshape((1, -1))
    
    m = df.iloc[CFG.N:, :].mean(axis=0).values
    
    if i in [0,1,2,3,4,5]:
        s = np.maximum(df.iloc[CFG.N:, :].std(axis=0), 1e-6).values
    elif i in [7, 8, 9]:
        s = np.maximum(df.iloc[CFG.N:, :].std(axis=0), 1e-7).values
    elif i == 12:
        s = df.iloc[CFG.N:, :].std(axis=0).values
        s[12:29] = 1e-10
    else:
        s = df.iloc[CFG.N:, :].std(axis=0).values
    
    df = (df - m) / s 
    df.fillna(0, inplace=True)
    df = df.astype("float32")
    
    norm_data.iloc[CFG.start_num[i]-1:CFG.end_num[i]-1, 0] = m
    norm_data.iloc[CFG.start_num[i]-1:CFG.end_num[i]-1, 1] = s
    
    for n in range(10):
        df.iloc[CFG.N*n:CFG.N*(n+1), :].to_parquet(f"{CFG.MAIN_FOLDER}data/data_{i}_{n}.parquet")
    
norm_data.to_parquet(f"{CFG.MAIN_FOLDER}data/norm_data.parquet")