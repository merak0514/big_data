import time
import numpy as np
import sys
import datetime
import pandas as pd
import os

date2position = {}
datestr2dateint = {}
str2int = {}
for i in range(24):
    str2int[str(i).zfill(2)] = i
for i in range(182):
    date = datetime.date(day=1, month=10, year=2018) + datetime.timedelta(days=i)
    date_int = int(date.__str__().replace("-", ""))
    date2position[date_int] = [i % 7, i // 7]
    datestr2dateint[str(date_int)] = date_int


def visit2array(table):
    strings = table[1]
    init = np.zeros((7, 26, 24))
    for string in strings:
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            x, y = date2position[datestr2dateint[date]]
            for visit in visit_lst:
                init[x][y][str2int[visit]] += 1
    return init


def visit2array_test():
    file_names = os.listdir('../../test/')
    length = len(file_names)
    start_time = time.time()
    for index, file_name in enumerate(file_names):
        if file_name.find('txt') == -1:
            continue
        table = pd.read_table("../../test/" + file_name + ".txt", header=None)
        array = visit2array(table)
        np.save("../../npy/test_visit/" + file_name + ".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs" % (time.time() - start_time))


def visit2array_train():
    # table = pd.read_csv("../data/train.txt", header=None)
    file_names = os.listdir('../../train/')
    length = len(file_names)
    start_time = time.time()
    for index, file_name in enumerate(file_names):
        if file_name.find('txt') == -1:
            continue
        table = pd.read_table("../../train/" + file_name + ".txt", header=None)
        array = visit2array(table)
        np.save("../../npy/train_visit/" + file_name + ".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs" % (time.time() - start_time))


def visit2array_valid():
    table = pd.read_csv("../../data/valid.txt", header=None)
    filenames = [a[0].split("/")[-1].split('.')[0] for a in table.values]
    length = len(filenames)
    start_time = time.time()
    for index, filename in enumerate(filenames):
        table = pd.read_table("../../data/train/" + filename + ".txt", header=None)
        array = visit2array(table)
        np.save("../../data/npy/train_visit/" + filename + ".npy", array)
        sys.stdout.write('\r>> Processing visit data %d/%d' % (index + 1, length))
        sys.stdout.flush()
    sys.stdout.write('\n')
    print("using time:%.2fs" % (time.time() - start_time))


if __name__ == '__main__':
    if not os.path.exists("../../npy/test_visit/"):
        os.makedirs("../../npy/test_visit/")
    if not os.path.exists("../../npy/train_visit/"):
        os.makedirs("../../npy/train_visit/")
    visit2array_train()
    # visit2array_valid()
    # visit2array_test()
