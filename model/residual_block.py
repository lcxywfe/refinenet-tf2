import tensorflow as tf


class BottleNeck(tf.keras.Model):
    def __init__(self, filter_num, stride=1, shortcut=False):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.shortcut = shortcut
        if self.shortcut:
            self.shortcut_conv = tf.keras.layers.Conv2D(filters=filter_num * 4,
                                                       kernel_size=(1, 1),
                                                       strides=stride)
            self.shortcut_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=True):

        if self.shortcut:
            residual = self.shortcut_conv(inputs)
            residual = self.shortcut_bn(residual)
        else:
            residual = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride, shortcut=True))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block

