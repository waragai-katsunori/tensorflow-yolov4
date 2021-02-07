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

        outputs = []
        stride = 5 + self.metalayer.classes
        for n, mask in enumerate(self.metalayer.mask):
            # x, y, w, h, o, c0, c1, ...
            xy_index = n * stride
            wh_index = xy_index + 2
            obj_index = xy_index + 4
            next_xy_index = xy_index + stride

            # x, y
            xy = sig[..., xy_index:wh_index]
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
            wh = x[..., wh_index:obj_index]
            wh = K.exp(wh) * anchor

            # o, c0, c1, ...
            oc = sig[..., obj_index:next_xy_index]

            outputs.append(K.concatenate([xy, wh, oc], axis=-1))

        return K.concatenate(outputs, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape
