# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 10:56
# @File     : histogram_equal.py
# @Software : PyCharm
from src.dehaze2 import deHaze
import cv2
import numpy as np
from matplotlib import pyplot as plt


def histogram_equal(m):
    g = m[:, :, 0]
    b = m[:, :, 1]
    r = m[:, :, 2]

    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)

    im_1 = im.copy()
    im_1[:, :, 0] = g
    im_1[:, :, 1] = b
    im_1[:, :, 2] = r
    return im_1


if __name__ == '__main__':

    im = cv2.imread('../../train/001/000024_001.jpg')
    cv2.imshow("before", im)
    im2 = deHaze(im).astype(np.uint8)
    print(type(im))
    print(type(im2))
    cv2.imshow('raw dehaze', im2)

    # split g,b,r
    g = im[:,:,0]
    b = im[:,:,1]
    r = im[:,:,2]
    # split g,b,r
    g2 = im2[:,:,0]
    b2 = im2[:,:,1]
    r2 = im2[:,:,2]

    print(type(r[0, 0]))
    print(type(r2[0, 0]))
    # Histogram Equalization
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)

    # Histogram Equalization
    r2 = cv2.equalizeHist(r2)
    g2 = cv2.equalizeHist(g2)
    b2 = cv2.equalizeHist(b2)

    im_1 = im.copy()
    im_1[:,:,0] = g
    im_1[:,:,1] = b
    im_1[:,:,2] = r

    im_2 = im2.copy()
    im_2[:,:,0] = g2
    im_2[:,:,1] = b2
    im_2[:,:,2] = r2

    print(im_1)

    cv2.imshow("after", im_1)
    cv2.imshow("dehaze+after", im_2)
    cv2.waitKey(0)