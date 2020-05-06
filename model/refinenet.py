import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ReLU, Add, MaxPool2D, UpSampling2D, BatchNormalization

import numpy as np

from .resnet import resnet_101

kern_init = keras.initializers.he_normal()
kern_reg = keras.regularizers.l2(1e-5)


class ResidualConvUnit(Model):
    def __init__(self, n_filters=256, kernel_size=3, name=''):
        super().__init__()
        self.relu1 = ReLU(name=name+'relu1')
        self.conv1 = Conv2D(n_filters, kernel_size, padding='same', name=name+'conv1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.relu2 = ReLU(name=name+'relu2')
        self.conv2 = Conv2D(n_filters, kernel_size, padding='same', name=name+'conv2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.add = Add(name=name+'sum')

    def call(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.relu2(y)
        y = self.conv2(y)
        return self.add([y, x])


class ChainedResidualPooling(Model):
    def __init__(self, n_filters=256, name=''):
        super().__init__()
        self.relu1 = ReLU(name=name+'relu')

        self.conv1 = Conv2D(n_filters, 3, padding='same', name=name+'conv1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool1', data_format='channels_last')

        self.conv2 = Conv2D(n_filters, 3, padding='same', name=name+'conv2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool2', data_format='channels_last')

        self.conv3 = Conv2D(n_filters, 3, padding='same', name=name+'conv3', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool3', data_format='channels_last')

        self.conv4 = Conv2D(n_filters, 3, padding='same', name=name+'conv4', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn4 = BatchNormalization()
        self.pool4 = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name=name+'pool4', data_format='channels_last')

        self.add = Add(name=name+'sum')

    def call(self, x, training=True):
        y0 = self.relu1(x)

        y1 = self.conv1(y0)
        y1 = self.bn1(y1, training=training)
        y1 = self.pool1(y1)

        y2 = self.conv2(y1)
        y2 = self.bn2(y2, training=training)
        y2 = self.pool2(y2)

        y3 = self.conv3(y2)
        y3 = self.bn3(y3, training=training)
        y3 = self.pool3(y3)

        y4 = self.conv4(y3)
        y4 = self.bn4(y4, training=training)
        y4 = self.pool4(y4)

        return self.add([y0, y1, y2, y3, y4])


class MultiResolutionFusion(Model):
    def __init__(self, n_filters=256, name=''):
        super().__init__()
        self.conv_low = Conv2D(n_filters, 3, padding='same', name=name+'conv_lo', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn_low = BatchNormalization()
        self.up_low = UpSampling2D(size=2, interpolation='bilinear', name=name+'up')

        self.conv_high = Conv2D(n_filters, 3, padding='same', name=name+'conv_hi', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn_high = BatchNormalization()

        self.add = Add(name=name+'sum')

    def call(self, high, low, training=True):
        low = self.conv_low(low)
        low = self.bn_low(low, training=training)
        low = self.up_low(low)

        high = self.conv_high(high)
        high = self.bn_high(high, training=training)

        return self.add([low, high])


class RefineBlock(Model):
    def __init__(self, n_high=256, n_low=256, block=0, with_low=True):
        super().__init__()
        if not with_low:
            self.rcu1 = ResidualConvUnit(n_filters=512, name='rb_{}_rcu_h1_'.format(block))
            self.rcu2 = ResidualConvUnit(n_filters=512, name='rb_{}_rcu_h2_'.format(block))
            self.fuse_pooling = ChainedResidualPooling(n_filters = 512, name='rb_{}_crp_'.format(block))
            self.rcu_out = ResidualConvUnit(n_filters = 512, name='rb_{}_rcu_o1_'.format(block))
        else:
            self.rcu_high1 = ResidualConvUnit(n_filters = n_high, name='rb_{}_rcu_h1_'.format(block))
            self.rcu_high2 = ResidualConvUnit(n_filters = n_high, name='rb_{}_rcu_h2_'.format(block))

            self.rcu_low1 = ResidualConvUnit(n_filters = n_low, name='rb_{}_rcu_l1_'.format(block))
            self.rcu_low2 = ResidualConvUnit(n_filters = n_low, name='rb_{}_rcu_l2_'.format(block))

            self.fuse = MultiResolutionFusion(n_filters = 256, name = 'rb_{}_mrf_'.format(block))
            self.fuse_pooling = ChainedResidualPooling(n_filters = 256, name='rb_{}_crp_'.format(block))
            self.rcu_out = ResidualConvUnit(n_filters = 256, name='rb_{}_rcu_o1_'.format(block))

    def call(self, high, low, training=True):
        if low is None:
            high = self.rcu1(high)
            high = self.rcu2(high)
            high = self.fuse_pooling(high, training=training)
            return self.rcu_out(high)
        else:
            high = self.rcu_high1(high)
            high = self.rcu_high2(high)

            low = self.rcu_low1(low)
            low = self.rcu_low2(low)

            fuse = self.fuse(high, low, training=training)
            fuse = self.fuse_pooling(fuse, training=training)

            return self.rcu_out(fuse)


def get_layer_by_name(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer


class RefineNet(Model):
    def __init__(self, num_class):
        super().__init__()
        self.resnet101 = resnet_101()

        self.conv0 = Conv2D(512, 1, padding='same', name='resnet_map1', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn0 = BatchNormalization()
        self.conv1 = Conv2D(256, 1, padding='same', name='resnet_map2', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(256, 1, padding='same', name='resnet_map3', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(256, 1, padding='same', name='resnet_map4', kernel_initializer=kern_init, kernel_regularizer=kern_reg)
        self.bn3 = BatchNormalization()

        self.rb0 = RefineBlock(block=4, with_low=False)
        self.rb1 = RefineBlock(n_low=512, block=3)
        self.rb2 = RefineBlock(block=2)
        self.rb3 = RefineBlock(block=1)

        self.rc0 = ResidualConvUnit(name='rf_rcu_o1_')
        self.rc1 = ResidualConvUnit(name='rf_rcu_o2_')

        self.up = UpSampling2D(size=4, interpolation='bilinear', name='rf_up_o')
        self.conv_out = Conv2D(num_class, 1, activation = 'softmax', name='rf_pred')

    def call(self, x, training=True):
        high = self.resnet101(x, training=training)
        low = [None, None, None]

        high[0] = self.conv0(high[0])
        high[0] = self.bn0(high[0], training=training)
        high[1] = self.conv1(high[1])
        high[1] = self.bn1(high[1], training=training)
        high[2] = self.conv2(high[2])
        high[2] = self.bn2(high[2], training=training)
        high[3] = self.conv3(high[3])
        high[3] = self.bn3(high[3], training=training)

        low[0] = self.rb0(high[0], None, training=training)
        low[1] = self.rb1(high[1], low[0], training=training)
        low[2] = self.rb2(high[2], low[1], training=training)
        y = self.rb3(high[3], low[2], training=training)

        y = self.rc0(y)
        y = self.rc1(y)

        y = self.up(y)
        y = self.conv_out(y)

        return y

