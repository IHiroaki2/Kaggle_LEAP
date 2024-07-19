#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import polars as pl
import random

import jax
import keras

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

from config import CFG


@keras.saving.register_keras_serializable()
class Conv1D(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__()

        self.hidden_size1 = CFG.hidden_size1
        self.hidden_size2 = CFG.hidden_size2
        self.hidden_size3 = CFG.hidden_size3
        self.kernel_size = CFG.kernel_size
        self.activation = CFG.activation
        self.dropout = CFG.conv1d_dropout

        self.conv1d_1 = tf.keras.layers.Conv1D(self.hidden_size1, self.kernel_size, padding='same', activation="linear")
        self.conv1d_2 = tf.keras.layers.Conv1D(self.hidden_size2, self.kernel_size, padding='same', activation="linear")
        self.conv1d_3 = tf.keras.layers.Conv1D(self.hidden_size3, self.kernel_size, padding='same', activation="linear")

        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        if self.activation == "relu":
            self.act_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
            self.act_2 = tf.keras.layers.Activation(tf.keras.activations.relu)
            self.act_3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        elif self.activation == "gelu":
            self.act_1 = tf.keras.layers.Activation(tf.keras.activations.gelu)
            self.act_2 = tf.keras.layers.Activation(tf.keras.activations.gelu)
            self.act_3 = tf.keras.layers.Activation(tf.keras.activations.gelu)
        else:
            raise ValueError


        self.dropout_1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout)
        # self.dropout_3 = tf.keras.layers.Dropout(self.dropout)

        self.pooling = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)

    def call(self, x):

        x0 = x
        x = self.conv1d_1(x)
        x = self.layernorm_1(x)
        x = self.act_1(x)
        x = self.dropout_1(x)
        x = self.conv1d_2(x)
        x = self.layernorm_2(x)
        x = self.act_2(x)
        x = self.dropout_2(x)
        x = self.conv1d_3(x)
        x = self.layernorm_3(x + x0 + self.pooling(x))
        x = self.act_3(x)

        return x




@keras.saving.register_keras_serializable()
class MultiHeadsAttention(tf.keras.Model):
    def __init__(self, num_heads=4, key_dim=256, dropout=0.15, transpose=True, *args, **kwargs):
        super(MultiHeadsAttention, self).__init__()

        self.transpose = transpose
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs):
        x = inputs[0]
        x0 = inputs[1]
        if self.transpose:
            x = tf.keras.ops.transpose(x, (0, 2, 1))
            x0 = tf.keras.ops.transpose(x0, (0, 2, 1))
            x_ = self.mha(x, x0, x0)
            x_ = self.layernorm(x_ + x)
            x0 = tf.keras.ops.transpose(x_, (0, 2, 1))
            x = self.dropout(x0)
            return x, x0

        else:
            x_ = self.mha(x, x0, x0)
            x0 = self.layernorm(x_ + x)
            x = self.dropout(x0)
            return x, x0



@keras.saving.register_keras_serializable()
class FFNN(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(FFNN, self).__init__()

        self.activation = CFG.activation

        self.hidden_size1 = CFG.hidden_size1
        self.hidden_size2 = CFG.hidden_size2
        self.hidden_size3 = CFG.hidden_size3

        self.linear_1 = tf.keras.layers.Dense(self.hidden_size1, use_bias=True, activation="linear")
        self.linear_2 = tf.keras.layers.Dense(self.hidden_size2, use_bias=True, activation="linear")
        self.linear_3 = tf.keras.layers.Dense(self.hidden_size3, use_bias=True, activation="linear")

        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


        if self.activation == "relu":
            self.act_1 = tf.keras.layers.Activation(tf.keras.activations.relu)
            self.act_2 = tf.keras.layers.Activation(tf.keras.activations.relu)
            self.act_3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        elif self.activation == "gelu":
            self.act_1 = tf.keras.layers.Activation(tf.keras.activations.gelu)
            self.act_2 = tf.keras.layers.Activation(tf.keras.activations.gelu)
            self.act_3 = tf.keras.layers.Activation(tf.keras.activations.gelu)
        else:
            raise ValueError

        self.pooling = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)

    def call(self, x):

        x0 = x

        x = self.linear_1(x)
        x = self.layernorm_1(x)
        x = self.act_1(x)

        x = self.linear_2(x)
        x = self.layernorm_2(x)
        x = self.act_2(x)

        x = self.linear_3(x)
        x = self.layernorm_3(x + x0 + self.pooling(x))
        x = self.act_3(x)

        return x



@keras.saving.register_keras_serializable()
class InputConv1D(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(InputConv1D, self).__init__()

        self.conv1d = tf.keras.layers.Conv1D(CFG.hidden_size3, 1, padding='same')

    def call(self, x):
        x = self.conv1d(x)

        return x


@keras.saving.register_keras_serializable()
class OutputTranspose(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(OutputTranspose, self).__init__()
        
        self.linear = tf.keras.layers.Dense(14)


    def call(self, x):

        p_all = self.linear(x)

        p_seq = p_all[:, :, :6]
        p_seq = tf.keras.ops.transpose(p_seq, (0, 2, 1))
        p_seq = tf.keras.layers.Flatten()(p_seq)
        assert p_seq.shape[-1] == 360

        p_flat = p_all[:, :, 6:6 + 8]
        p_flat = tf.keras.ops.mean(p_flat, axis=1)
        assert p_flat.shape[-1] == 8

        p = tf.keras.ops.concatenate([p_seq, p_flat], axis=1)

        return p


@keras.saving.register_keras_serializable()
class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(CustomModel, self).__init__()

        self.num_heads = CFG.num_heads
        self.key_dim1 = CFG.key_dim1
        self.key_dim2 = CFG.key_dim2
        self.dropout = CFG.mha_dropout

        self.input_conv1d = InputConv1D()

        self.conv1d_01 = Conv1D()
        self.conv1d_02 = Conv1D()
        self.conv1d_03 = Conv1D()
        self.conv1d_04 = Conv1D()
        self.conv1d_05 = Conv1D()
        self.conv1d_06 = Conv1D()
        self.conv1d_07 = Conv1D()
        self.conv1d_08 = Conv1D()
        self.conv1d_09 = Conv1D()
        self.conv1d_10 = Conv1D()

        self.mha_01 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim1, dropout=self.dropout, transpose=True)
        self.mha_02 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim1, dropout=self.dropout, transpose=False)
        self.mha_03 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim2, dropout=self.dropout, transpose=True)
        self.mha_04 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim2, dropout=self.dropout, transpose=False)
        self.mha_05 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim1, dropout=self.dropout, transpose=True)
        self.mha_06 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim1, dropout=self.dropout, transpose=False)
        self.mha_07 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim2, dropout=self.dropout, transpose=True)
        self.mha_08 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim2, dropout=self.dropout, transpose=False)
        self.mha_09 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim1, dropout=self.dropout, transpose=True)
        self.mha_10 = MultiHeadsAttention(num_heads=self.num_heads, key_dim=self.key_dim1, dropout=self.dropout, transpose=False)

        self.ffnn = FFNN()

        self.output_transpose = OutputTranspose()


    def x_to_seq(self, x):
        x_seq0 = tf.keras.ops.transpose(tf.keras.ops.reshape(x[:, 0:60 * 6], (-1, 6, 60)), (0, 2, 1))
        x_seq1 = tf.keras.ops.transpose(tf.keras.ops.reshape(x[:, 60 * 6 + 16:60 * 9 + 16], (-1, 3, 60)), (0, 2, 1))
        x_flat = tf.keras.ops.reshape(x[:, 60 * 6:60 * 6 + 16], (-1, 1, 16))
        x_flat = tf.keras.ops.repeat(x_flat, 60, axis=1)

        return tf.keras.ops.concatenate([x_seq0, x_seq1, x_flat], axis=-1)


    def call(self, x):

        x = self.x_to_seq(x)
        x = x0 = self.input_conv1d(x)

        x = self.conv1d_01(x)
        x, x1 = self.mha_01([x, x0])
        x = self.conv1d_02(x)
        x, x2 = self.mha_02([x, x0])

        x = self.conv1d_03(x)
        x, x1 = self.mha_03([x, x1])
        x = self.conv1d_04(x)
        x, x2 = self.mha_04([x, x2])

        x = self.conv1d_05(x)
        x, x1 = self.mha_05([x, x1])
        x = self.conv1d_06(x)
        x, x2 = self.mha_06([x, x2])

        x = self.conv1d_07(x)
        x, x1 = self.mha_07([x, x1])
        x = self.conv1d_08(x)
        x, x2 = self.mha_08([x, x2])

        x = self.conv1d_09(x)
        x, x1 = self.mha_09([x, x1])
        x = self.conv1d_10(x)
        x, x2 = self.mha_10([x, x2])

        x = self.ffnn(x)

        p = self.output_transpose(x)

        return p