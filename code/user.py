# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 10:38
# @File     : user.py
# @Software : PyCharm
import os
from collections import defaultdict
import re

train_path = '../train/'
test_path = '../test/'
save_file = '../train_users.txt'
# save_path = '../users/'
# if os.path.exists(save_path):
#     os.mkdir(save_path)
files_name = os.listdir(train_path)
users = defaultdict(lambda: 0)
for i, file in enumerate(files_name):
    if file.find('txt') == -1:
        continue
    with open(train_path+file) as f:
        for line in f:
            user = re.findall('(.+)?\t', line)[0]
            users[user] += 1
    f.close()
    print('finish one', i, file)
    print(len(users))
print('finish all files')
sorted_users = sorted(users.items(), key=lambda item: item[1], reverse=True)
print(sorted_users)
with open(save_file, 'w+') as ff:
    for i in sorted_users:
        ff.write(str(i))

f.close()
