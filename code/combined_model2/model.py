# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 9:26
# @File     : model.py
# @Software : PyCharm
from src.resnet import *
from src.visit_network import *
from keras.models import Model

VISIT_INPUT_SHAPE = np.array([7*24])
IMAGE_INPUT_SHAPE = (100, 100, 3)


def image_net():
    image_input = Input(shape=IMAGE_INPUT_SHAPE, name='image_input')
    x = resnet_v2(image_input, trainable=True)
    outputs = Dense(9, activation='softmax', name='image_softmax')(x)
    model = Model(inputs=image_input, outputs=outputs)
    return model


def visit_net():
    visit_input = Input(shape=VISIT_INPUT_SHAPE, name='visit_input')
    x = dense_net(visit_input, trainable=True)
    outputs = Dense(9, activation='softmax', name='visit_softmax')(x)
    model = Model(inputs=visit_input, outputs=outputs)
    return model


def combined_net():
    image_input = Input(shape=IMAGE_INPUT_SHAPE, name='image_input')
    image_net_ = resnet_v2(image_input, trainable=False)
    visit_input = Input(shape=VISIT_INPUT_SHAPE, name='visit_input')
    # visit_input_ = Reshape((7*24,))(visit_input)
    print(visit_input.shape)
    # print(visit_input_.shape)
    visit_net_ = dense_net(visit_input, trainable=False)
    x = keras.layers.concatenate([image_net_, visit_net_])
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(9, activation='softmax')(x)
    model = Model(inputs=[image_input, visit_input], outputs=outputs)
    return model
