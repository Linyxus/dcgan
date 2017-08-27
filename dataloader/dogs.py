#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Linyxus'

from scipy.io import loadmat

def read_list(filename):
    data = loadmat(filename)
    print(data)


if __name__ == '__main__':
    read_list('../dogs/file_list.mat')