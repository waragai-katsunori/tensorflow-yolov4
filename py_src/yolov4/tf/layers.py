"""
MIT License

Copyright (c) 2020-2021 Hyeonki Hong <hhk7734@gmail.com>

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
from tensorflow import keras


class Mish(keras.layers.Layer):
    def call(self, x):
        # pylint: disable=no-self-use
        return x * keras.backend.tanh(keras.backend.softplus(x))


class YOLOConv2D(keras.Sequential):
    def __init__(
        self,
        activation: str,
        batch_normalize: int,
        filters: int,
        kernel_regularizer,
        padding: int,
        size: int,
        stride: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if stride == 2:
            self.add(keras.layers.ZeroPadding2D(((1, 0), (1, 0))))

        self.add(
            keras.layers.Conv2D(
                filters=filters,
                kernel_size=(size, size),
                padding="same" if stride == 1 else "valid",
                strides=(stride, stride),
                use_bias=not batch_normalize,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.constant_initializer(0.0),
            )
        )

        if batch_normalize:
            self.add(keras.layers.BatchNormalization())

        if activation == "mish":
            self.add(Mish())
        elif activation == "leaky":
            self.add(keras.layers.LeakyReLU(alpha=0.1))
        elif activation == "relu":
            self.add(keras.layers.ReLU())
        elif activation == "linear":
            pass
        else:
            raise ValueError(f"YOLOConv2D: '{activation}' is not supported.")
