# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:19
# @File     : visit_net.py
# @Software : PyCharm
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ReLU, Flatten, Reshape, Dropout
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
