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

train_path = '../train/'
test_path = '../test/'
record_path = '../record.csv'
weights_save_path = '../save.h5'
BATCH_SIZE = 256
os.environ['CUDA_VISIBLE_DEVICES'] = "3,6"


def load_train_data(X_path=train_path, y_path=record_path):
    folders_name_ = os.listdir(X_path)
    folders_name = []
    for folder in folders_name_:
        # print(folder.find('txt'))
        if folder.find('txt') == -1:
            folders_name.append(os.path.join(train_path, folder) + '/')
    print(folders_name)

    X_name = []
    X_train_ = []
    y_train_ = []
    X_eval_ = []
    y_eval_ = []
    for i in range(len(folders_name)):
        folder = str(i+1).zfill(3) + '/'
        print(folder)
        files_name = os.listdir(folder)
        label = np.zeros(9, dtype=np.int)
        label[i] = 1
        for j, file in enumerate(files_name):
            name = re.findall('([0-9]+)_', file)[0]
            image = cv2.imread(os.path.join(folder, file))
            image = np.array(image, np.int)
            X_name.append(name)
            if j % 5 == 0:
                y_eval_.append(label)
                X_eval_.append(image)
            else:
                y_train_.append(label)
                X_train_.append(image)
        print('finish', str(i))
    X_name = np.array(X_name, np.str)
    X_train_ = np.array(X_train_, np.int)
    y_train_ = np.array(y_train_, np.int)
    X_eval_ = np.array(X_eval_, np.int)
    y_eval_ = np.array(y_eval_, np.int)
    print('X shape', X_train_.shape)
    print('y shape', y_train_.shape)
    print('X eval shape', X_eval_.shape)
    print('y eval shape', y_eval_.shape)
    return X_train_, y_train_, X_eval_, y_eval_


def train():
    X_train, y_train, X_eval, y_eval = load_train_data()
    datagen = ImageDataGenerator(rotation_range=50, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                                 zoom_range=[0.8, 1.2], horizontal_flip=True)
    datagen.fit(X_train)

    resnet = ResNet50(weights=None, input_shape=(100, 100, 3), classes=9)
    resnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train, y_train, X_eval, y_eval = load_train_data()
    datagen = ImageDataGenerator(rotation_range=50, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                                 zoom_range=[0.8, 1.2], horizontal_flip=True, vertical_flip=True,
                                 brightness_range=[0.8, 1.2])
    datagen.fit(X_train)

    resnet = ResNet50(weights=None, input_shape=(100, 100, 3), classes=9)
    resnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('../model_keras.h5', monitor='val_loss', save_best_only=True,
                                 save_weights_only=True)
    csv_logger = CSVLogger('../cnn_log.csv', separator=',', append=False)
    es = EarlyStopping(patience=10, restore_best_weights=True)
    tb = TensorBoard()
    # resnet.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=10000, validation_data=(X_eval, y_eval),
    #            callbacks=[checkpoint, csv_logger, tb, es])
    resnet.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                         steps_per_epoch=int(len(X_train) / BATCH_SIZE) * 3, epochs=200,
                         validation_data=(X_eval, y_eval),
                         callbacks=[checkpoint, csv_logger, tb, es],
                         class_weight=[0.9, 1, 2, 4, 2, 1.2, 2, 2.5, 2.5])

    resnet.save_weights(weights_save_path)

    score = resnet.evaluate(X_train, y_train, batch_size=256)
    print('train acc', score)

    classes_counts = np.zeros(9)
    corrects_counts = np.zeros(9)
    labels = np.argmax(y_eval, axis=1)
    predicts = resnet.predict(X_eval, batch_size=256)
    predicts = np.argmax(predicts, axis=1)
    corrects = labels == predicts
    for i in range(len(corrects)):
        classes_counts[labels[i]] += 1
        corrects_counts[labels[i]] += corrects[i]
    acc_by_classes = corrects_counts / classes_counts
    print('acc_by_classes: ', acc_by_classes)

    score = resnet.evaluate(X_eval, y_eval, batch_size=256)
    print('eval acc', score)
    checkpoint = ModelCheckpoint('../model_keras.h5', monitor='val_loss', save_best_only=True,
                                 save_weights_only=True)
    csv_logger = CSVLogger('../cnn_log.csv', separator=',', append=False)
    es = EarlyStopping(patience=10, restore_best_weights=True)
    tb = TensorBoard()
    # resnet.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=10000, validation_data=(X_eval, y_eval),
    #            callbacks=[checkpoint, csv_logger, tb, es])
    resnet.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                         steps_per_epoch=int(len(X_train) / BATCH_SIZE) * 3, epochs=200,
                         validation_data=(X_eval, y_eval),
                         callbacks=[checkpoint, csv_logger, tb, es])

    resnet.save_weights(weights_save_path)

    print('eval acc:')
    evaluate(X_eval, y_eval, model=resnet)
    print('train acc')
    evaluate(X_train, y_train, model=resnet)


def evaluate(X_eval, y_eval, weights_path=weights_save_path, model=None):
    if not (weights_path or model):
        print('eval wrong!')
        assert 0
    if weights_path:
        model = ResNet50(weights=None, input_shape=(100, 100, 3), classes=9)
        model.load_weights(weights_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classes_counts = np.zeros(9)  # 每个区域有几个
    corrects_counts = np.zeros(9)  # 判斷對了的有幾個
    labels = np.argmax(y_eval, axis=1)
    predicts = model.predict(X_eval, batch_size=128)
    predicts = np.argmax(predicts, axis=1)
    corrects = labels == predicts
    for i in range(len(corrects)):
        classes_counts[labels[i]] += 1
        corrects_counts[labels[i]] += corrects[i]
    acc_by_classes = corrects_counts / classes_counts
    for i, j in enumerate(acc_by_classes):
        print('class: ', i, ', acc: ', j)
    print('total_acc', sum(corrects_counts) / sum(classes_counts))


def predict(data_path=test_path, weights_path=weights_save_path, model=None):
    def load_data(path_=test_path):
        images = []
        for index in range(10000):
            image = cv2.imread(os.path.join(path_, str(index).zfill(6)+'.jpg'))
            image = np.array(image, dtype=np.int)
            images.append(image)

        images = np.array(images, dtype=np.int)
        print('finish loading unpredicted data')
        return images

    predict_path = 'predict.txt'
    data = load_data(data_path)
    if not (weights_path or model):
        print('eval wrong!')
        assert 0
    if weights_path:
        model = ResNet50(weights=None, input_shape=(100, 100, 3), classes=9)
        model.load_weights(weights_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predicts = model.predict(data, batch_size=32)
    predicts = np.argmax(predicts, axis=1) + 1
    print(predicts)
    with open(predict_path, 'w+') as f:
        print('start writing predict...')
        for i, result in enumerate(predicts):
            image_id = str(i).zfill(6)
            result = str(np.str(result)).zfill(3)
            f.write('\t'.join([image_id, result]))
            f.write('\n')
        f.close()
    print('finish predicting.')


if __name__ == '__main__':
    train()
    predict()
