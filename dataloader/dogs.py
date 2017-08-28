#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Linyxus'

from scipy.io import loadmat
import scipy.misc as smi
import numpy as np

img_size = (32, 32)

def read_list(filename):
    data = loadmat(filename)['file_list']
    return [x[0][0] for x in data]

def read_img(filename):
    im = smi.imread(filename)
    return smi.imresize(im, img_size)

def load(path, verbose=1):
    fl = read_list('%s/file_list.mat' % path)[:5000]
    tot = len(fl)
    ims = []
    if verbose:
        print('[INFO] Loading images from %s' % path)
    for i, f in enumerate(fl):
        im = read_img('%s/Images/%s' % (path, f))
        if verbose:
            print('(%d/%d) - %.2f%% Load %s : %s' % (i+1, tot, (i+1) / tot * 100, f, im.shape))
        ims.append(im)
    ims = np.array(ims)
    ims = ims / 255
    # due to the feature of cifar10, model is designed to receive channel first images as input
    # thus, change the format
    ims = ims.swapaxes(3, 2)
    ims = ims.swapaxes(2, 1)
    if verbose:
        print('[INFO] Loaded. ( shape:', ims.shape, ')')
    return ims

if __name__ == '__main__':
    imgs = load('../dogs')
    print(imgs[0])