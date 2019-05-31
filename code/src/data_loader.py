# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 9:15
# @File     : data_loader.py
# @Software : PyCharm
import sys
sys.path.append('../')
import numpy as np
import re
import os
import cv2
from src.dehaze2 import deHaze
from src.histogram_equal import histogram_equal
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout
from keras.models import Sequential
IMAGE_TRAIN_PATH = '../../train/'
IMAGE_TEST_PATH = '../../test/'
VISIT_TRAIN_PATH = '../../npy/train_visit/'
VISIT_TEST_PATH = '../../npy/test_visit/'
WEIGHTS_SAVE_PATH = '../../result/save_visit_only.h5'
PREDICT_PATH = '../../result/predict_visit_only.txt'
MODEL_CKPT = '../../result/model_visit_only.h5'


def load_train_data(image_path=IMAGE_TRAIN_PATH, visit_path=VISIT_TRAIN_PATH, output_shape=True, vision=1):
    folders_name_ = os.listdir(image_path)
    folders_name = []
    for folder in folders_name_:
        # print(folder.find('txt'))
        if folder.find('txt') == -1:
            folders_name.append(os.path.join(IMAGE_TRAIN_PATH, folder) + '/')
    print(folders_name)

    X_name = []
    X_train_image_ = []
    X_train_visit_ = []
    y_train_ = []

    X_eval_image_ = []
    X_eval_visit_ = []
    y_eval_ = []
    for i in range(len(folders_name)):
        folder = str(i + 1).zfill(3) + '/'
        folder = image_path + folder
        print(folder)
        files_name = os.listdir(folder)
        label = np.zeros(9, dtype=np.int)
        label[i] = 1
        for j, file in enumerate(files_name):
            name = re.findall('^(.+)\.jpg', file)[0]
            image = cv2.imread(os.path.join(folder, file))
            image = deHaze(image)  # 去雾
            # image = histogram_equal(image)  # 降噪  保留一个即可
            shape = image.shape

            npy_name = '.'.join([name, 'txt', 'npy'])
            npy_datum = np.load(visit_path + npy_name)
            if vision == 1:  # 第一种：直接取平均
                visit = np.ndarray.flatten(np.average(npy_datum, axis=1))
            elif vision == 2:  # 第二种：后面做一个一维卷积
                visit = np.transpose(npy_datum, [0, 2, 1])
                visit = visit.reshape([7*24, 26])

            X_name.append(name)
            if j % 5 == 0:
                image = np.array(image, np.int)
                y_eval_.append(label)
                X_eval_visit_.append(visit)
                X_eval_image_.append(image)
            else:
                M = cv2.getRotationMatrix2D(
                    (shape[0] / 2, shape[1] / 2), np.random.randint(-10, 10), 1)
                image = cv2.warpAffine(image, M, dsize=(shape[0], shape[1]))
                image = np.array(image, np.int)
                y_train_.append(label)
                X_train_visit_.append(visit)
                X_train_image_.append(image)
                y_train_.append(label)
                X_train_visit_.append(visit)
                X_train_image_.append(np.flip(image, axis=np.random.randint(-1, 2)))
        print('finish', str(i + 1))
    X_name = np.array(X_name, np.str)
    X_train_image_ = np.array(X_train_image_, np.float)
    X_train_visit_ = np.array(X_train_visit_, np.float)
    y_train_ = np.array(y_train_, np.float)
    X_eval_image_ = np.array(X_eval_image_, np.float)
    X_eval_visit_ = np.array(X_eval_visit_, np.float)
    y_eval_ = np.array(y_eval_, np.float)
    if output_shape:
        print('X train image shape', X_train_image_.shape)
        print('X train visit shape', X_train_visit_.shape)
        print('y train shape', y_train_.shape)
        print('X eval image shape', X_eval_image_.shape)
        print('X eval visit shape', X_eval_visit_.shape)
        print('y eval shape', y_eval_.shape)
    return X_train_image_, X_train_visit_, y_train_, X_eval_image_, X_eval_visit_, y_eval_


def load_test_data_visit(visit_path=VISIT_TEST_PATH):
    visits_ = []
    for index in range(10000):
        name = str(index).zfill(6)

        npy_datum = np.load(os.path.join(visit_path, name + '.txt.npy'))
        visit = np.ndarray.flatten(np.average(npy_datum, axis=1))
        visits_.append(visit)
    visits_ = np.array(visits_, dtype=np.float)
    print('visit shape', visits_.shape)
    print('finish loading unpredicted data')
    return visits_


def load_test_data(image_path=IMAGE_TEST_PATH, visit_path=VISIT_TEST_PATH):
    images_ = []
    visits_ = []
    for index in range(10000):
        name = str(index).zfill(6)
        image = cv2.imread(os.path.join(image_path, name + '.jpg'))
        image = np.array(image, dtype=np.int)
        image = deHaze(image)  # 去雾
        # image = histogram_equal(image)  # 降噪  保留一个即可
        images_.append(image)

        npy_datum = np.load(os.path.join(visit_path, name+'.txt.npy'))
        visit = np.ndarray.flatten(np.average(npy_datum, axis=1))
        visits_.append(visit)

    images_ = np.array(images_, dtype=np.int)
    visits_ = np.array(visits_, dtype=np.float)
    print('image shape', images_.shape)
    print('visit shape', visits_.shape)
    print('finish loading unpredicted data')
    return images_, visits_
