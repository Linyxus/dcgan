#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import keras
keras.backend.set_image_data_format('channels_first')

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

def plot_images(images, dim=4):
    plt.figure()
    for i in range(images.shape[0]):
        plt.subplot(dim, dim, i+1)
        img = images[i]
        # to channel last
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_images(images, idx, dim=4):
    plt.figure()
    for i in range(images.shape[0]):
        plt.subplot(dim, dim, i+1)
        img = images[i]
        # to channel last
        img = np.swapaxes(img, 0, 1)
        img = np.swapaxes(img, 1, 2)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('dcgan_%s.jpg' % idx)

class DCGAN(object):
    def __init__(self, data):
        self.channel = data.shape[1]
        self.row = data.shape[2]
        self.col = data.shape[3]
        self.real_images = data
        self.init_d()
        self.init_g()
        self.init_DM()
        self.init_AM()

    def init_d(self):
        print('[INFO] Init discriminator...')

        self.D = Sequential()
        depth = 128
        dropout = 0.4
        input_shape = (self.channel, self.row, self.col)

        self.D.add(Conv2D(depth, 2, strides=2, padding='same', input_shape=input_shape))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 2, 2, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 4, 2, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 8, 2, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
        self.D.add(Dense(1, activation='sigmoid'))
        self.D.summary()

    def init_g(self):
        print('[INFO] Init generator...')

        self.G = Sequential()
        depth = 256
        dropout = 0.4
        dim = 8

        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((depth, dim, dim)))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth / 16), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(3, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()

    def init_DM(self):
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.D)
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def init_AM(self):
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.G)
        self.AM.add(self.D)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,
                        metrics=['accuracy'])

    def train(self, steps=10000, batch_size=256, save_interval=100):
        self.logger = {
            'a': {
                'loss': [],
                'acc': []
            },
            'd': {
                'loss': [],
                'acc': []
            }
        }
        for i in range(steps):
            real_img = self.real_images[np.random.randint(0, self.real_images.shape[0], size=batch_size)]
            # noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            noise = np.random.normal(0.0, 1.0, size=(batch_size, 100))
            fake_img = self.G.predict(noise)
            x = np.concatenate((real_img, fake_img))
            y = np.ones((2*batch_size, 1))
            y[batch_size:, :] = 0
            d_loss = self.DM.train_on_batch(x, y)

            # x = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            x = np.random.normal(0.0, 1.0, size=(batch_size, 100))
            y = np.ones((batch_size, 1))
            a_loss = self.AM.train_on_batch(x, y)

            print('(%d/%d) [D loss: %f, acc: %f] [A loss: %f, acc: %f]' % (i+1, steps, d_loss[0], d_loss[1],
                                                                           a_loss[0], a_loss[1]))
            self.logger['d']['loss'].append(d_loss[0])
            self.logger['d']['acc'].append(d_loss[1])
            self.logger['a']['loss'].append(a_loss[0])
            self.logger['a']['acc'].append(a_loss[1])
            if (i + 1) % save_interval == 0:
                self.save_fake(i+1)

    def plot_real(self, sample=9, dim=3):
        x = np.random.randint(0, self.real_images.shape[0], size=sample)
        plot_images(self.real_images[x], dim=dim)

    def plot_fake(self, sample=9, dim=3):
        # noise = np.random.uniform(-1.0, 1.0, size=(sample, 100))
        noise = np.random.normal(0.0, 1.0, size=(sample, 100))
        images = self.G.predict(noise)
        plot_images(images, dim=dim)

    def save_fake(self, idx, sample=9, dim=3):
        # noise = np.random.uniform(-1.0, 1.0, size=(sample, 100))
        noise = np.random.normal(0.0, 1.0, size=(sample, 100))
        images = self.G.predict(noise)
        save_images(images, idx, dim=dim)

    def save_log(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.logger, f, protocol=pickle.HIGHEST_PROTOCOL)
