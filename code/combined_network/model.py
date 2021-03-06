# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 15:42
# @File     : picture_classification.py
# @Software : PyCharm
from src.resnet import *
from src.visit_network import *
from keras.models import Model


VISIT_INPUT_SHAPE = np.array([7*24])
IMAGE_INPUT_SHAPE = (100, 100, 3)


def combined_net():
    image_input = Input(shape=IMAGE_INPUT_SHAPE, name='image_input')
    image_net = resnet_v2(image_input)
    visit_input = Input(shape=VISIT_INPUT_SHAPE, name='visit_input')
    # visit_input_ = Reshape((7*24,))(visit_input)
    print(visit_input.shape)
    # print(visit_input_.shape)
    visit_net = visit_net_v1(visit_input)
    x = keras.layers.concatenate([image_net, visit_net])
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(9, activation='softmax')(x)
    model = Model(inputs=[image_input, visit_input], outputs=outputs)
    return model

