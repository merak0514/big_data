# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 15:42
# @File     : picture_classification.py
# @Software : PyCharm
# image model
import numpy as np
import cv2
import re
import os
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

IMAGE_TRAIN_PATH = '../../train/'
IMAGE_TEST_PATH = '../../test/'
WEIGHTS_SAVE_PATH = '../../save.h5'
MODEL_CKPT = '../../result/model_keras.h5'
BATCH_SIZE = 128
os.environ['CUDA_VISIBLE_DEVICES'] = "9"


def load_train_data(image_path=IMAGE_TRAIN_PATH):
    folders_name_ = os.listdir(image_path)
    folders_name = []
    for folder in folders_name_:
        # print(folder.find('txt'))
        if folder.find('txt') == -1:
            folders_name.append(os.path.join(IMAGE_TRAIN_PATH, folder) + '/')
    print(folders_name)

    X_name = []
    X_train_ = []
    y_train_ = []
    X_eval_ = []
    y_eval_ = []
    for i in range(len(folders_name)):
        folder = str(i+1).zfill(3) + '/'
        folder = image_path + folder
        print(folder)
        files_name = os.listdir(folder)
        label = np.zeros(9, dtype=np.int)
        label[i] = 1
        for j, file in enumerate(files_name):
            name = re.findall('^([0-9]+)_', file)[0]
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
                                 zoom_range=[0.8, 1.2], horizontal_flip=True, vertical_flip=True,
                                 brightness_range=[0.8, 1.2])
    datagen.fit(X_train)

    resnet = ResNet50(weights=None, input_shape=(100, 100, 3), classes=9)
    resnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_CKPT, monitor='val_loss', save_best_only=True,
                                 save_weights_only=True)
    # csv_logger = CSVLogger('../cnn_log.csv', separator=',', append=False)
    es = EarlyStopping(patience=10, restore_best_weights=True)
    tb = TensorBoard()
    # resnet.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=10000, validation_data=(X_eval, y_eval),
    #            callbacks=[checkpoint, csv_logger, tb, es])
    resnet.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                         steps_per_epoch=int(len(X_train) / BATCH_SIZE) * 3, epochs=200,
                         validation_data=(X_eval, y_eval),
                         callbacks=[checkpoint, tb, es],
                         class_weight=[0.9, 1, 2, 4, 2, 1.2, 2, 2.5, 2.5])

    resnet.save_weights(WEIGHTS_SAVE_PATH)

    print('eval acc:')
    evaluate(X_eval, y_eval, model=resnet)
    print('train acc')
    evaluate(X_train, y_train, model=resnet)


def evaluate(X_eval_image=None, y_eval=None, weights_path=WEIGHTS_SAVE_PATH, model=None):
    if not (weights_path or model):
        print('eval wrong!')
        assert 0
    if weights_path:
        model = ResNet50(weights=None, input_shape=(100, 100, 3), classes=9)
        model.load_weights(weights_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if X_eval_image == None:
        X_eval_image, y_eval, _, _ = load_train_data()

    labels = np.argmax(y_eval, axis=1)
    predicts = model.predict(X_eval_image, batch_size=128)
    predicts = np.argmax(predicts, axis=1)
    corrects = labels == predicts

    recall_counts = np.zeros(9)  # 每个区域有几个，真實的總的A類
    recall_corrects_counts = np.zeros(9)  # 預測正確的A類, 用于計算查全率（預測正確的A類/真實的總的A類）

    precision_counts = np.zeros(9)  # 預測的A類總數，用於計算查準率（預測正確的A類/預測的A類總數）
    precision_correct_counts = np.zeros(9)  # 預測正確的A類

    for i in range(len(corrects)):
        recall_counts[labels[i]] += 1
        recall_corrects_counts[labels[i]] += corrects[i]

        precision_counts[predicts[i]] += 1
        precision_correct_counts[predicts[i]] += corrects[i]

    recall_by_classes = recall_corrects_counts / recall_counts
    precision_by_classes = precision_correct_counts / precision_counts
    for i, j in enumerate(recall_by_classes):
        print('class: ', i, ' recall (the origin acc): ', j, 'precision: ', precision_by_classes[i])
    print('total_acc', sum(recall_corrects_counts) / sum(recall_counts))


def predict(image_path=IMAGE_TEST_PATH, weights_path=WEIGHTS_SAVE_PATH, model=None):
    def load_data(path_=IMAGE_TEST_PATH):
        images = []
        for index in range(10000):
            image = cv2.imread(os.path.join(path_, str(index).zfill(6)+'.jpg'))
            image = np.array(image, dtype=np.int)
            images.append(image)

        images = np.array(images, dtype=np.int)
        print('finish loading unpredicted data')
        return images

    predict_path = 'predict.txt'
    data = load_data(image_path)
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
    evaluate()
