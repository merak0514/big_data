# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:34
# @File     : visit_only.py
# @Software : PyCharm
# visit model
import numpy as np
import re
import os
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from keras.layers import Dense,BatchNormalization, LeakyReLU, Dropout
from keras.models import Sequential

IMAGE_TRAIN_PATH = '../../train/'
IMAGE_TEST_PATH = '../../test/'
VISIT_TRAIN_PATH = '../../npy/train_visit/'
VISIT_TEST_PATH = '../../npy/test_visit/'
WEIGHTS_SAVE_PATH = '../../result/save_visit_only.h5'
PREDICT_PATH = '../../result/predict_visit_only.txt'
MODEL_CKPT = '../../result/model_visit_only.h5'
BATCH_SIZE = 256
os.environ['CUDA_VISIBLE_DEVICES'] = "9"


def load_train_data(image_path=IMAGE_TRAIN_PATH, visit_path=VISIT_TRAIN_PATH, output_shape=True):

    X_name = []
    X_train_visit_ = []
    y_train_ = []

    X_eval_visit_ = []
    y_eval_ = []
    for i in range(9):
        folder = str(i + 1).zfill(3) + '/'
        folder = image_path + folder
        print(folder)
        files_name = os.listdir(folder)
        label = np.zeros(9, dtype=np.int)
        label[i] = 1
        for j, file in enumerate(files_name):
            name = re.findall('^(.+)\.jpg', file)[0]

            npy_name = '.'.join([name, 'txt', 'npy'])
            npy_datum = np.load(visit_path + npy_name)
            visit = np.ndarray.flatten(np.average(npy_datum, axis=1))

            X_name.append(name)
            if j % 5 == 0:
                y_eval_.append(label)
                X_eval_visit_.append(visit)
            else:
                y_train_.append(label)
                X_train_visit_.append(visit)
        print('finish', str(i + 1))
    X_name = np.array(X_name, np.str)
    X_train_visit_ = np.array(X_train_visit_, np.float)
    y_train_ = np.array(y_train_, np.float)
    X_eval_visit_ = np.array(X_eval_visit_, np.float)
    y_eval_ = np.array(y_eval_, np.float)
    if output_shape:
        print('X train visit shape', X_train_visit_.shape)
        print('y train shape', y_train_.shape)
        print('X eval visit shape', X_eval_visit_.shape)
        print('y eval shape', y_eval_.shape)
    return X_train_visit_, y_train_, X_eval_visit_, y_eval_


class Model(Sequential):
    def __init__(self):
        features = 7*24
        layers = [Dense(128, input_dim=features), LeakyReLU(0.05), Dropout(0.2),
                  # Dense(128), BatchNormalization(), LeakyReLU(0.05),
                  # Dense(128), BatchNormalization(), LeakyReLU(0.05), Dropout(0.2),
                  Dense(128), BatchNormalization(), LeakyReLU(0.05),
                  Dense(128), BatchNormalization(), LeakyReLU(0.05), Dropout(0.2),
                  Dense(128), BatchNormalization(), LeakyReLU(0.05),
                  Dense(256), BatchNormalization(), LeakyReLU(0.05), Dropout(0.3),
                  Dense(256), BatchNormalization(), LeakyReLU(0.05),
                  Dense(256), BatchNormalization(), LeakyReLU(0.05), Dropout(0.3),
                  Dense(256), BatchNormalization(), LeakyReLU(0.05),
                  Dense(256), BatchNormalization(), LeakyReLU(0.05), Dropout(0.3),
                  Dense(256), BatchNormalization(), LeakyReLU(0.05),
                  # Dense(256), LeakyReLU(0.05), Dropout(0.3),
                  # Dense(256), LeakyReLU(0.05), Dropout(0.3),
                  Dense(512), BatchNormalization(), LeakyReLU(0.05), Dropout(0.4),
                  Dense(512), BatchNormalization(), LeakyReLU(0.05),
                  Dense(512), BatchNormalization(), LeakyReLU(0.05), Dropout(0.4),
                  Dense(512), BatchNormalization(), LeakyReLU(0.05),
                  # Dense(512), LeakyReLU(0.05), Dropout(0.4),
                  # Dense(1024), LeakyReLU(0.05), Dropout(0.5),
                  Dense(1024), BatchNormalization(), LeakyReLU(0.05), Dropout(0.5),
                  Dense(256), BatchNormalization(), LeakyReLU(0.05),
                  Dense(9, activation='softmax')
                  ]
        Sequential.__init__(self, layers=layers)

        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.summary()

    def eval(self, X_eval_visit=None, y_eval=None, weights_path=WEIGHTS_SAVE_PATH):
        if not (X_eval_visit or y_eval):
            X_train_visit, y_train, X_eval_visit, y_eval = load_train_data()

        if weights_path:
            self.load_weights(weights_path)
            self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        labels = np.argmax(y_eval, axis=1)
        predicts = model.predict(X_eval_visit, batch_size=128)
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

    def predict_(self, visit_path=VISIT_TEST_PATH, model_path=None,
                 X=None, predict_path=PREDICT_PATH):
        def load_data():
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

        if model_path:
            self.load_weights(model_path)
        if not X:
            X = load_data()
        result = self.predict(X)
        predicts = np.argmax(result, axis=1) + 1
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

    def train(self, save_path=WEIGHTS_SAVE_PATH, batch_size=BATCH_SIZE):
        X_train, y_train, X_eval, y_eval = load_train_data()

        checkpoint = ModelCheckpoint(MODEL_CKPT, monitor='val_acc', save_best_only=True,
                                     save_weights_only=True)
        es = EarlyStopping(patience=20, restore_best_weights=True)
        # tb = TensorBoard()

        self.fit(X_train, y_train, batch_size=batch_size, epochs=10000, validation_data=(X_eval, y_eval),
                 callbacks=[checkpoint, es])
        self.save_weights(save_path)
        score = self.evaluate(X_train, y_train, batch_size=10000)
        print('train loss', score)

        score = self.evaluate(X_eval, y_eval, batch_size=10000)
        print('eval loss', score)

        print('eval acc:')
        self.eval(X_eval, y_eval, weights_path=None)
        print('train acc')
        self.eval(X_train, y_train, weights_path=None)

        self.predict_()


if __name__ == '__main__':
    model = Model()
    model.eval()
