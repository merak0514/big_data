# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 9:26
# @File     : model.py
# @Software : PyCharm
from src.resnet import *
from src.visit_network import *
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential

VISIT_INPUT_SHAPE = np.array([7*24])
VISIT_INPUT_SHAPE_2 = np.array([7*24, 26])
IMAGE_INPUT_SHAPE = (100, 100, 3)
RESNET_WEIGHT_PATH = '../../result/resnet50.h5'


def image_net_1():
    """自己的resnet50"""
    image_input = Input(shape=IMAGE_INPUT_SHAPE, name='image_input')
    x = resnet_v2(image_input, trainable=True)
    outputs = Dense(9, activation='softmax', name='image_softmax')(x)
    model = Model(inputs=image_input, outputs=outputs)
    return model


def visit_net_1():
    visit_input = Input(shape=VISIT_INPUT_SHAPE, name='visit_input')
    x = visit_net_v1(visit_input, trainable=True)
    outputs = Dense(9, activation='softmax', name='visit_softmax')(x)
    model = Model(inputs=visit_input, outputs=outputs)
    return model


def combined_net_0():
    image_input = Input(shape=IMAGE_INPUT_SHAPE, name='image_input')
    image_net_ = resnet_v2(image_input, trainable=False)
    visit_input = Input(shape=VISIT_INPUT_SHAPE, name='visit_input')
    # visit_input_ = Reshape((7*24,))(visit_input)
    print(visit_input.shape)
    # print(visit_input_.shape)
    visit_net_ = visit_net_v1(visit_input, trainable=False)
    x = keras.layers.concatenate([image_net_, visit_net_])
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(9, activation='softmax')(x)
    model = Model(inputs=[image_input, visit_input], outputs=outputs)
    return model


def combined_net_1():
    image_net_ = ResNet50(weights=RESNET_WEIGHT_PATH, input_shape=IMAGE_INPUT_SHAPE, classes=9, include_top=False)
    for layer in image_net_.layers:
        layer.trainable = False
    new_image_output = GlobalAveragePooling2D(name='avg_pool')(image_net_.layers[-1].output)
    new_image_output = Dense(256, activation='relu', name='combine_dense_1')(new_image_output)

    visit_input = Input(shape=VISIT_INPUT_SHAPE, name='visit_input')
    # visit_input_ = Reshape((7*24,))(visit_input)
    # print(visit_input_.shape)
    visit_net_ = visit_net_v1(visit_input, trainable=False)

    x = keras.layers.concatenate([new_image_output, visit_net_])
    x = Dense(1024, name='combine_dense_2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(9, activation='softmax', name='combine_dense_3')(x)
    model = Model(inputs=[image_net_.inputs[0], visit_input], outputs=outputs)
    # model.summary()
    return model


def image_net_2():
    """官方resnet"""
    resnet = ResNet50(weights=RESNET_WEIGHT_PATH, input_shape=(100, 100, 3), classes=9, include_top=False)
    resnet.trainable = False
    resnet_end = resnet.layers[-1].output
    new_end = GlobalAveragePooling2D(name='avg_pool')(resnet_end)
    new_end = Dense(9, activation='softmax')(new_end)
    i_net = Model(inputs=resnet.inputs, outputs=new_end)
    i_net.set_weights(resnet.get_weights())
    return i_net


def visit_net_2():
    visit_input = Input(shape=VISIT_INPUT_SHAPE_2, name='visit_input')
    x = visit_net_v2(visit_input, trainable=True)
    outputs = Dense(9, activation='softmax', name='visit_softmax')(x)
    model = Model(inputs=visit_input, outputs=outputs)
    return model


def combined_net_2():
    image_net_ = ResNet50(weights=RESNET_WEIGHT_PATH, input_shape=IMAGE_INPUT_SHAPE, classes=9, include_top=False)
    for layer in image_net_.layers:
        layer.trainable = False
    new_image_output = GlobalAveragePooling2D(name='avg_pool')(image_net_.layers[-1].output)
    new_image_output = Dense(256, activation='relu', name='combine_dense_1')(new_image_output)

    visit_input = Input(shape=VISIT_INPUT_SHAPE_2, name='visit_input')
    # visit_input_ = Reshape((7*24,))(visit_input)
    # print(visit_input_.shape)
    visit_net_ = visit_net_v2(visit_input, trainable=False)

    x = keras.layers.concatenate([new_image_output, visit_net_])
    x = Dense(1024, name='combine_dense_2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(9, activation='softmax', name='combine_dense_3')(x)
    model = Model(inputs=[image_net_.inputs[0], visit_input], outputs=outputs)
    # model.summary()
    return model


if __name__ == '__main__':
    combined_net_2()
