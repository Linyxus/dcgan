#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Linyxus'

import pickle
import numpy as np

def _load_batch(filename):
    dict = {}
    f = open(filename, 'rb')
    data = pickle.load(f, encoding='bytes')
    f.close()

    imgs = np.array(data[b'data'])
    imgs = imgs.reshape((imgs.shape[0], 3, 32, 32))
    dict['img'] = imgs

    labels = np.array(data[b'labels'])
    dict['label'] = labels

    return dict

def _load_meta(filename):
    f = open(filename, 'rb')
    data = pickle.load(f, encoding='bytes')
    f.close()
    meta = data[b'label_names']
    meta = [x.decode('ascii') for x in meta]
    return meta

def load(path):
    dict = {}
    data = _load_batch('%s/data_batch_1' % path)
    imgs, labels = data['img'], data['label']
    for x in [2, 3, 4, 5]:
        filename = '%s/data_batch_%d' % (path, x)
        data = _load_batch(filename)
        imgs = np.concatenate((imgs, data['img']))
        labels = np.concatenate((labels, data['label']))
    dict['img'] = imgs / 255
    dict['label'] = labels
    dict['meta'] = _load_meta('%s/batches.meta' % path)
    return dict

if __name__ == '__main__':
    data = load('../cifar10')
    print(data)
    print(data['img'].shape, data['label'].shape, data['meta'])
