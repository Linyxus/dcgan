#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dataloader.cifar10
import dcgan

if __name__ == '__main__':
    print('*** DCGAN trained with cifar10 ***')
    data = dataloader.cifar10.load()['img']
    model = dcgan.DCGAN(data)
    try:
        model.train(steps=3000)
    except Exception as e:
        print(e)
    finally:
        model.save_log('train_log.pickle')
        model.G.save('generator.h5')
