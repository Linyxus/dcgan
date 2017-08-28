#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataloader.cifar10
import dataloader.dogs
import dcgan

def train_cifar10():
    print('*** DCGAN trained with cifar10 ***')
    data = dataloader.cifar10.load('cifar10')['img']
    model = dcgan.DCGAN(data)
    try:
        model.train(steps=3000)
    except Exception as e:
        print(e)
    finally:
        model.save_log('train_log.pickle')
        model.G.save('generator.h5')

def train_dogs():
    print('*** DCGAN trained with dogs ***')
    data = dataloader.dogs.load('dogs')
    model = dcgan.DCGAN(data)
    try:
        model.train(steps=10000, save_interval=500)
    except Exception as e:
        print(e)
    finally:
        model.save_log('train_log.pickle')
        model.G.save('generator.h5')

if __name__ == '__main__':
    train_dogs()
    # train_cifar10()
