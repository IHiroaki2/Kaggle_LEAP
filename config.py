#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math


class CFG:

    # MAIN_FOLDER = ""

    ### data_splitting.py
    N = 1009152

    start_num = [1, 61, 121, 181, 241, 301, 361, 377, 437, 497, 557, 617, 677, 737, 797, 857, 917]
    end_num = [61, 121, 181, 241, 301, 361, 377, 437, 497, 557, 617, 677, 737, 797, 857, 917, 925]


    ### download_from_hf.py 
    repo_id = "LEAP/ClimSim_low-res"
    local_dir = f"nc_data"


    ### myfunctions.py
    TRAIN_COLS = ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v", "state_ps",
                  "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX", "pbuf_TAUX", "pbuf_TAUY", "pbuf_COSZRS",
                  "cam_in_ALDIF", "cam_in_ALDIR", "cam_in_ASDIF", "cam_in_ASDIR", "cam_in_LWUP", "cam_in_ICEFRAC",
                  "cam_in_LANDFRAC", "cam_in_OCNFRAC", "cam_in_SNOWHLAND", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O"]
    TRAIN_COLS_60LEVEL = ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v", "pbuf_ozone", "pbuf_CH4", "pbuf_N2O"]
    TARGET_COLS = ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v","cam_out_NETSW","cam_out_FLWDS",
                   "cam_out_PRECSC","cam_out_PRECC","cam_out_SOLS","cam_out_SOLL","cam_out_SOLSD","cam_out_SOLLD"]
    TARGET_COLS_60LEVEL = ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v"]

    seed = 42
    batch_size = 1024*4

    
    ## training.py
    training = "kaggle"  ## "kaggle" "additional", "12files"
    trial = "001"
    summary = True
    target_ncols = 368
    
    retrain = False
    trained_model_filepath = f"model/"
    

    ### Prediction
    pred_model_filepath = f"model/leap_model_001_2024-07-18_18-26.keras"


    ### model.py
    #Conv1D & FFNN
    hidden_size1 = 256
    hidden_size2 = 128
    hidden_size3 = 64
    kernel_size = 4
    conv1d_dropout = 0.15
    activation = "gelu"

    #Attention
    num_heads = 2
    key_dim1 = 64
    key_dim2 = 16
    mha_dropout = 0.15

    ### Learning Rate
    epochs = 50
    learning_rate = 8e-4
    epochs_warmup = 2
    epochs_ending = 0
    warmup_rate = 1e-5
    alpha = 0.01

    trained_epochs = 0
    pi = math.pi

    if training == "kaggle":
        steps_per_epoch = int(np.ceil(9284198 / batch_size))
    elif training == "additional":
        steps_per_epoch = int(np.ceil(22487040 / batch_size))
    elif training == "12files":
        steps_per_epoch = int(np.ceil(1210982 / batch_size))
    else:
        raise ValueError

    
    ### Train
    monitor = "val_loss"
    loss = "mse"
    model_folderpath = f"model"