# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:48
# @File     : train.py
# @Software : PyCharm
# combine model
from model import combined_net
import numpy as np
import cv2
import re
import os
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Reshape

IMAGE_TRAIN_PATH = '../../train/'
IMAGE_TEST_PATH = '../../test/'
VISIT_TRAIN_PATH = '../../npy/train_visit/'
VISIT_TEST_PATH = '../../npy/test_visit/'
WEIGHTS_SAVE_PATH = '../../result/save_combine.h5'
MODEL_CKPT = '../../result/model_keras_combine.h5'
BATCH_SIZE = 64
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


def load_train_data(image_path=IMAGE_TRAIN_PATH, visit_path=VISIT_TRAIN_PATH, output_shape=True):
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
            shape = image.shape

            npy_name = '.'.join([name, 'txt', 'npy'])
            npy_datum = np.load(visit_path + npy_name)
            visit = np.ndarray.flatten(np.average(npy_datum, axis=1))

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


def generate_generator_multiple(generator1, generator2, dir1, dir2, y, batch_size):
    genX1 = generator1.flow(dir1, y,
                            batch_size=batch_size,
                            shuffle=False,
                            seed=7)

    genX2 = generator2.flow(dir2, y,
                            batch_size=batch_size,
                            shuffle=False,
                            seed=7)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label


def train():
    X_train_image, X_train_visit, y_train, X_eval_image, X_eval_visit, y_eval = load_train_data()
    # datagen1 = ImageDataGenerator(rotation_range=50, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
    #                               zoom_range=[0.8, 1.2], horizontal_flip=True)
    # datagen2 = ImageDataGenerator()
    # datagen = generate_generator_multiple(datagen1, datagen2, X_train_image, X_train_visit,
    # y_train, batch_size=BATCH_SIZE)

    model = combined_net()
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(MODEL_CKPT, monitor='val_loss', save_best_only=True,
                                 save_weights_only=True)
    # csv_logger = CSVLogger('../cnn_log.csv', separator=',', append=False)
    es = EarlyStopping(patience=10, restore_best_weights=True)
    # tb = TensorBoard()
    model.fit([X_train_image, X_train_visit], y_train, batch_size=BATCH_SIZE, epochs=10000, shuffle=True,
              validation_data=([X_eval_image, X_eval_visit], y_eval),
              callbacks=[checkpoint, es],
              class_weight=[0.9, 1, 2, 4, 2, 1.2, 2, 2.5, 2.5])
    # model.fit_generator(datagen,
    #                     steps_per_epoch=int(len(X_train_image) / BATCH_SIZE) * 3, epochs=200,
    #                     validation_data=([X_eval_image, X_eval_visit], y_eval),
    #                     callbacks=[checkpoint, csv_logger, tb, es],
    #                     class_weight=[0.9, 1, 2, 4, 2, 1.2, 2, 2.5, 2.5])

    model.save_weights(WEIGHTS_SAVE_PATH)
    print('weight saved')

    score = model.evaluate([X_train_image, X_train_visit], y_train, batch_size=BATCH_SIZE)
    print('train acc', score)

    print('eval acc:')
    evaluate(X_eval_image, X_eval_visit, y_eval, model=model)
    print('train acc')
    evaluate(X_train_image, X_train_visit, y_train, model=model)


def evaluate(X_eval_image=None, X_eval_visit=None, y_eval=None, weights_path=WEIGHTS_SAVE_PATH, model=None):
    if not X_eval_image:
        X_train_image, X_train_visit, y_train, X_eval_image, X_eval_visit, y_eval = load_train_data()

    if not (weights_path or model):
        print('eval wrong!')
        assert 0
    if weights_path:
        model = combined_net()
        model.load_weights(weights_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    labels = np.argmax(y_eval, axis=1)
    predicts = model.predict([X_eval_image, X_eval_visit], batch_size=128)
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


def predict(image_path=IMAGE_TEST_PATH, visit_path=VISIT_TEST_PATH, weights_path=WEIGHTS_SAVE_PATH, model=None,
            predict_path='predict_combine.txt'):
    def load_data():
        images_ = []
        visits_ = []
        for index in range(10000):
            name = str(index).zfill(6)
            image = cv2.imread(os.path.join(image_path, name + '.jpg'))
            image = np.array(image, dtype=np.int)
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

    images, visits = load_data()
    if not (weights_path or model):
        print('eval wrong!')
        assert 0
    if weights_path:
        model = combined_net()
        model.load_weights(weights_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    predicts = model.predict([images, visits], batch_size=32)
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
    predict()
