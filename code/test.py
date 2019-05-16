# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 11:56
# @File     : test.py
# @Software : PyCharm
import csv
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import cv2
# p = cv2.imread('../train/001/000009_001.jpg')
# print(p.shape)
# cv2.imshow('p', p)
# cv2.waitKey()
# with open('../train/001/000009_001.jpg') as p:
#     print(p)
#     plt.imshow(p)
#     plt.show()


# file = open('users.txt', 'r')
# new_file = open('users.csv', 'w+')
# c_writer = csv.writer(new_file)
# c_writer.writerow(['user_id', 'count'])
#
# for line in file:
#     count = 0
#     users = line.split(')(')
#     users[0] = users[0][1:]
#     users[-1] = users[-1][:-1]
#     for u in users:
#         tmp = u.split(', ')
#         tmp[0] = tmp[0][1: -1]
#         c_writer.writerow(tmp)
#         count += 1
#         if count % 1000 == 0:
#             print('finish 1000', count)
#
# new_file.close()
# print('finish')


file = open('users.csv', 'r')
next(file)

count = defaultdict(lambda: 0)
tmp_count = 0
for line in file:
    datum = line.split(',')
    count[int(datum[1])] += 1
    tmp_count += 1
    if tmp_count % 10000 == 0:
        print('finish 10000', tmp_count)

print(count)
np.save('count_users.npy', count)
