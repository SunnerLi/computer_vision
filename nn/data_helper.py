import numpy as np
import preprocess
import cv2
import os

"""
POS_IMG_SITE = './img/pos/'
NEG_IMG_SITE = './img/neg/'
"""
TEST_IMG_SITE = './img/test/'

POS_IMG_SITE = './work/pos/'
NEG_IMG_SITE = './work/neg/'

if not os.path.exists('./work') == True:
    preprocess.preProcess()

def load():
    imgs = []
    tags = []
    for name in os.listdir(POS_IMG_SITE):
        imgs.append(cv2.imread(POS_IMG_SITE + name, 0))
        tags.append(1)
    for name in os.listdir(NEG_IMG_SITE):
        imgs.append(cv2.imread(NEG_IMG_SITE + name, 0))
        tags.append(0)
    return np.asarray(imgs, dtype=float), tags

def load_test():
    imgs = []
    tags = []
    for name in os.listdir(TEST_IMG_SITE):
        tag = int(name.split('-')[1].split('.')[0])
        imgs.append(cv2.imread(TEST_IMG_SITE + name, 0))
        tags.append(tag)
    return np.asarray(imgs, dtype=float), tags

def indexTo01(a, max_index=2):
    if a >= max_index:
        print "<error> index out of range"
        return
    result = [0] * max_index
    result[a] = 1
    return np.asarray(result)

load_test()