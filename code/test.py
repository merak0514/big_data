# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 11:56
# @File     : test.py
# @Software : PyCharm
from matplotlib import pyplot as plt
import cv2
p = cv2.imread('../train/001/000009_001.jpg')
print(p.shape)
cv2.imshow('p', p)
cv2.waitKey()
# with open('../train/001/000009_001.jpg') as p:
#     print(p)
#     plt.imshow(p)
#     plt.show()
