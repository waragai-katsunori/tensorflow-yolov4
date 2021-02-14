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
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class YoloLayer(Layer):
    def __init__(self, metalayer, metanet):
        super().__init__(name=metalayer.name)
        self.metalayer = metalayer
        self.metanet = metanet

        self.cx_cy = []
        for j in range(metalayer.height):
            self.cx_cy.append([])
            for i in range(metalayer.width):
                self.cx_cy[j].append([i, j])

    def call(self, x):
        """
        @param `x`: Dim(height, width, height, channels)

        @return: Dim(height, width, height, channels)
        """
        sig = K.sigmoid(x)
        sig_s = tf.split(sig, 3, axis=-1)

        raw_s = tf.split(x, 3, axis=-1)

        output = []
        for n, mask in enumerate(self.metalayer.mask):
            # # x, y, w, h, o, c0, c1, ...
            # Operation not supported on Edge TPU
            xy, _, oc = tf.split(sig_s[n], [2, 2, -1], axis=-1)
            _, wh, _ = tf.split(raw_s[n], [2, 2, -1], axis=-1)

            # Can be Mapped to Edge TPU
            # x, y
            if self.metalayer.scale_x_y != 1.0:
                xy = (xy - 0.5) * self.metalayer.scale_x_y + 0.5
            xy += self.cx_cy
            xy /= (self.metalayer.width, self.metalayer.height)

            # w, h
            anchor = self.metalayer.anchors[mask]
            anchor = (
                anchor[0] / self.metanet.width,
                anchor[1] / self.metanet.height,
            )
            wh = K.exp(wh) * anchor

            output.append(K.concatenate([xy, wh, oc], axis=-1))
        return K.concatenate(output, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape
