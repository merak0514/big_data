# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:19
# @File     : visit_net.py
# @Software : PyCharm
from keras.layers import Dense, BatchNormalization, Activation, ReLU, Flatten, Reshape, Dropout, Conv1D, MaxPooling1D
import numpy as np


def dense_net(inputs, trainable=True):
    x = Dense(1024, name='visit_0', trainable=trainable)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, name='visit_1', trainable=trainable)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, name='visit_2', trainable=trainable)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # x = Reshape([None, 256])(x)
    # print('x', x.shape)
    return x


def visit_net_v2(inputs, trainable=True):
    x = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
               trainable=trainable, name='visit_0')(inputs)  # 64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, padding='same')(x)  # 84

    x = Conv1D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
               trainable=trainable, name='visit_0')(x)  # 128
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, padding='same')(x)  # 42

    x = Conv1D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
               trainable=trainable, name='visit_0')(x)  # 256
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, padding='same')(x)  # 21

    x = Conv1D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
               trainable=trainable, name='visit_0')(x)  # 512
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2, padding='same')(x)  # 10

    x = Flatten()(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x
