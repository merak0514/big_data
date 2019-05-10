# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:19
# @File     : visit_net.py
# @Software : PyCharm
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ReLU, Flatten, Reshape, Dropout
from keras.backend import reshape
import numpy as np


def dense_net(inputs):
    x = Dense(1024)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # x = Reshape([None, 256])(x)
    # print('x', x.shape)
    return x
