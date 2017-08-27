#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Linyxus'

from scipy.io import loadmat

def read_list(filename):
    data = loadmat(filename)['file_list']
    return [x[0][0] for x in data]


if __name__ == '__main__':
    print(read_list('../dogs/file_list.mat'))