# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 9:21
# @File     : run_model.py
# @Software : PyCharm
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from src.data_loader import load_test_data, load_train_data
from combined_model2.model import *


IMAGE_TRAIN_PATH = '../../train/'
IMAGE_TEST_PATH = '../../test/'
VISIT_TRAIN_PATH = '../../npy/train_visit/'
VISIT_TEST_PATH = '../../npy/test_visit/'
WEIGHTS_SAVE_PATH_IMAGE = '../../result/save_combine_image.h5'
WEIGHTS_SAVE_PATH_VISIT = '../../result/save_combine_visit.h5'
WEIGHTS_SAVE_PATH = '../../result/save_combine.h5'
MODEL_CKPT_IMAGE = '../../result/model_keras_combine2_image.h5'
MODEL_CKPT_VISIT = '../../result/model_keras_combine2_visit.h5'
MODEL_CKPT_COMBINE = '../../result/model_keras_combine2.h5'
PREDICT_PATH = '../../result/predict_combine.txt'
MODEL_COMBINED_VERSION = 2
MODEL_IMAGE_VERSION = 2
MODEL_VISIT_VERSION = 2
BATCH_SIZE = 256
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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


def train(train_visit=True, train_image=True, load_ckpt_image=False, load_ckpt_visit=False):
    X_train_image, X_train_visit, y_train, X_eval_image, X_eval_visit, y_eval \
        = load_train_data(version=MODEL_VISIT_VERSION)
    es = EarlyStopping(patience=10, restore_best_weights=True)

    if train_visit:
        """start of visit part"""
        checkpoint = ModelCheckpoint(MODEL_CKPT_VISIT, monitor='val_acc', save_best_only=True, save_weights_only=True)
        model_visit = eval('visit_net_'+str(MODEL_VISIT_VERSION))()
        model_visit.summary()
        model_visit.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if load_ckpt_visit:
            model_visit.load_weights(MODEL_CKPT_VISIT)
            print('load ckpt visit successfully')

        model_visit.fit(X_train_visit, y_train, batch_size=BATCH_SIZE, epochs=10000,
                        validation_data=(X_eval_visit, y_eval),
                        callbacks=[checkpoint, es])
        model_visit.save_weights(WEIGHTS_SAVE_PATH_VISIT)
        """end of visit part"""

    if train_image:
        checkpoint = ModelCheckpoint(MODEL_CKPT_IMAGE, monitor='val_acc', save_best_only=True, save_weights_only=True)
        model_image = eval('image_net_'+str(MODEL_IMAGE_VERSION))()
        model_image.summary()
        # datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
        #                              zoom_range=[0.8, 1.2], brightness_range=[0.8, 1.2])
        # datagen.fit(X_train_image)

        model_image.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if load_ckpt_image:
            model_image.load_weights(MODEL_CKPT_IMAGE)
            print('load ckpt image successfully')

        # model_image.fit_generator(datagen.flow(X_train_image, y_train, batch_size=BATCH_SIZE),
        #                           steps_per_epoch=int(len(X_train_image) / BATCH_SIZE) * 3, epochs=500,
        #                           validation_data=(X_eval_image, y_eval),
        #                           callbacks=[checkpoint, es],
        #                           class_weight=[0.9, 1, 2, 4, 2, 1.2, 2, 2.5, 2.5])
        model_image.fit(X_train_image, y_train, batch_size=BATCH_SIZE, epochs=10000, callbacks=[es, checkpoint],
                        validation_data=(X_eval_image, y_eval), shuffle=True)
        model_image.save_weights(WEIGHTS_SAVE_PATH_IMAGE)
        """end of image part"""

    checkpoint = ModelCheckpoint(MODEL_CKPT_COMBINE, monitor='val_acc', save_best_only=True, save_weights_only=True)

    model = eval('combined_net_'+str(MODEL_COMBINED_VERSION))()

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights(WEIGHTS_SAVE_PATH_VISIT, by_name=True)
    model.load_weights(WEIGHTS_SAVE_PATH_IMAGE, by_name=True)

    model.fit([X_train_image, X_train_visit], y_train, batch_size=BATCH_SIZE, epochs=10000, shuffle=True,
              validation_data=([X_eval_image, X_eval_visit], y_eval),
              callbacks=[checkpoint, es])
    # class_weight = [0.9, 1, 2, 4, 2, 1.2, 2, 2.5, 2.5]
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
    if X_eval_image is None:
        X_train_image, X_train_visit, y_train, X_eval_image, X_eval_visit, y_eval\
            = load_train_data(version=MODEL_VISIT_VERSION)

    if not (weights_path or model):
        print('eval wrong!')
        assert 0
    if weights_path:
        model = eval('combined_net_'+str(MODEL_COMBINED_VERSION))()
        model.load_weights(weights_path)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    labels = np.argmax(y_eval, axis=1)
    predicts = model.predict([X_eval_image, X_eval_visit], batch_size=128)
    predicts = np.argmax(predicts, axis=1)
    corrects = labels == predicts

    cm = confusion_matrix(labels, predicts)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    print('cm', cm)
    print('cm_norm', cm_norm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm_norm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > cm_norm.max()/2 else "black")
    fig.tight_layout()
    plt.show()

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
            predict_path=PREDICT_PATH):

    images, visits = load_test_data(image_path, visit_path)
    if not (weights_path or model):
        print('eval wrong!')
        assert 0
    if weights_path:
        model = eval('combined_net_'+str(MODEL_COMBINED_VERSION))()
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
    train(train_visit=True, train_image=True, load_ckpt_image=False, load_ckpt_visit=False)
    evaluate()
    predict()
