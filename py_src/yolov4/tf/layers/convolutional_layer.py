"""
MIT License

Copyright (c) 2021 Hyeonki Hong <hhk7734@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import tensorflow as tf
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Activation,
    # BatchNormalization,
    Conv2D,
    LeakyReLU,
    ReLU,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import get_custom_objects


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        """
        training = False, trainable = False
        training = False, trainable = True
            gamma, beta, mean, and variance are fixedd
        training = True, trainable = False
            gamma, beta are fixed
            mean and variance are updated
        training = True, trainable = True
            gamma, beta, mean, and variance are updated

        When trying transfer learning, if you set trainable of backbone to
        `False`, you can freeze backbone.
        """
        if training and self.trainable:
            return super().call(x, training=True)
        return super().call(x, training=False)


class ConvolutionalLayer(Sequential):
    def __init__(self, metalayer, metanet):
        super().__init__(name=metalayer.name)
        self.metalayer = metalayer
        self.metanet = metanet

        if metalayer.stride == 2:
            self.add(ZeroPadding2D(((1, 0), (1, 0))))

        self.add(
            Conv2D(
                filters=metalayer.filters,
                kernel_size=metalayer.size,
                padding="same" if metalayer.stride == 1 else "valid",
                strides=metalayer.stride,
                use_bias=not metalayer.batch_normalize,
                kernel_regularizer=L2(l2=0.005),
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
            )
        )

        if metalayer.batch_normalize:
            self.add(
                BatchNormalization(epsilon=1e-5, momentum=self.metanet.momentum)
            )

        if metalayer.activation == "mish":
            self.add(Activation("mish"))
        elif metalayer.activation == "leaky":
            self.add(LeakyReLU(alpha=0.1))
        elif metalayer.activation == "relu":
            self.add(ReLU())
        elif metalayer.activation == "linear":
            pass
        else:
            raise ValueError(
                f"YOLOConv2D: '{metalayer.activation}' is not supported."
            )


"""
digantamisra98/Mish/Mish/TFKeras/mish.py

MIT License

Copyright (c) 2019 Diganta Misra
"""


class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super().__init__(activation, **kwargs)
        self.__name__ = "mish"


def mish(x):
    return x * K.tanh(K.softplus(x))


get_custom_objects().update({"mish": Mish(mish)})
