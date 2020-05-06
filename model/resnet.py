import tensorflow as tf
from .residual_block import make_bottleneck_layer


class ResNet(tf.keras.Model):
    def __init__(self, layer_params):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(7, 7),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        y1 = self.layer1(x, training=training)
        y2 = self.layer2(y1, training=training)
        y3 = self.layer3(y2, training=training)
        y4 = self.layer4(y3, training=training)

        return [y4, y3, y2, y1]


def resnet_101():
    return ResNet(layer_params=[3, 4, 23, 3])


