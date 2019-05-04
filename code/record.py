# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 10:38
# @Author   : Merak
# @File     : user.py
# @Software : PyCharm
import os
import csv
import re

train_path = '../train/'
test_path = '../test/'
record_path = '../record.csv'
folders_name_ = os.listdir(train_path)
folders_name = []
for folder in folders_name_:
    # print(folder.find('txt'))
    if folder.find('txt') == -1:
        folders_name.append(os.path.join(train_path, folder)+'/')
print(folders_name)

record = open(record_path, 'w+', newline='')
csv_writer = csv.writer(record)
for i, folder in enumerate(folders_name):
    files_name = os.listdir(folder)
    for file in files_name:
        name = re.findall('([0-9]+)_', file)[0]
        csv_writer.writerow([name, i])
    print('finish {}'.join(str(i)))

record.close()


# users = defaultdict(lambda: 0)
# for i, file in enumerate(folders_name):
#     with open(train_path+file) as f:
#         for line in f:
#             user = re.findall('(.+)?\t', line)[0]
#             users[user] += 1
#     f.close()
#     print('finish one', i)
# print('finish all files')
# sorted_users = sorted(users.items(), key=lambda item: item[1], reverse=True)
# print(sorted_users)
# with open(user, 'w+') as ff:
#     for i in sorted_users:
#         ff.write(str(i))
#
# f.close()
