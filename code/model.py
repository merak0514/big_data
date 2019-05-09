# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 15:42
# @File     : picture_classification.py
# @Software : PyCharm
import numpy as np
import cv2
import csv
import re
import os
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Reshape
from keras.backend import squeeze
from resnet import *
from visit_network import *
from keras.models import Model

train_path = '../train/'
record_path = '../record.csv'
weights_save_path = '../save.h5'
BATCH_SIZE = 256

VISIT_INPUT_SHAPE = np.array([7*24])
IMAGE_INPUT_SHAPE = (100, 100, 3)


def combined_net():
    image_input = Input(shape=IMAGE_INPUT_SHAPE, name='image_input')
    image_net = resnet_v2(image_input)
    visit_input = Input(shape=VISIT_INPUT_SHAPE, name='visit_input')
    # visit_input_ = Reshape((7*24,))(visit_input)
    print(visit_input.shape)
    # visit_input_ = squeeze(visit_input_, 1)
    # print(visit_input_.shape)
    visit_net = dense_net(visit_input)
    x = keras.layers.concatenate([image_net, visit_net])
    x = Dense(1024)(x)
    x = ReLU()(x)
    outputs = Dense(9, activation='softmax')(x)
    model = Model(inputs=[image_input, visit_input], outputs=outputs)
    return model

