#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from huggingface_hub import hf_hub_download
import time
import json

from config import CFG


for year in ["3", "4", "5", "6", "7", "8", "9"]:
    if year == "9":
        m = ["01"]
    else:
        m = ["01", "02", "03", "04" ,"05", "06", "07", "08", "09", "10", "11", "12"]

    for month in m:
        if month in ["01", "03", "05", "07", "08", "10", "12"]:
            days = 31
        elif month in ["04", "06", "09", "11"]:
            days = 30
        else:
            days = 28

        for day in range(1, days+1):
            for t in range(24):
                filename_i = f"train/000{year}-{month}/E3SM-MMF.mli.000{year}-{month}-{day:02d}-{t*3600:05d}.nc"
                filename_o = f"train/000{year}-{month}/E3SM-MMF.mlo.000{year}-{month}-{day:02d}-{t*3600:05d}.nc"
                hf_hub_download(repo_id=CFG.repo_id, filename=filename_i, repo_type="dataset", local_dir=CFG.local_dir)
                hf_hub_download(repo_id=CFG.repo_id, filename=filename_o, repo_type="dataset", local_dir=CFG.local_dir)
                if t == 0:
                    print(filename_i, filename_o)
                time.sleep(0.5)