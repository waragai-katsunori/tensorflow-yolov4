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

        self.stride = metalayer.classes + 5
        self.num_masks = len(metalayer.mask)

    def call(self, x):
        """
        @param `x`: Dim(height, width, height, channels)

        @return: Dim(height, width, height, channels)
            xy: logistic, scale
            wh: raw
            oc: logistic
        """

        output = tf.TensorArray(x.dtype, size=self.num_masks)
        scale_x_y = tf.constant(self.metalayer.scale_x_y, x.dtype)

        for n in tf.range(self.num_masks):
            xy_index = n * self.stride
            wh_index = xy_index + 2
            obj_index = xy_index + 4
            next_xy_index = (n + 1) * self.stride

            if scale_x_y == 1.0:
                xy = K.sigmoid(x[..., xy_index:wh_index])
            else:
                xy = scale_x_y * K.sigmoid(x[..., xy_index:wh_index]) - (
                    0.5 * (scale_x_y - 1)
                )

            oc = K.sigmoid(x[..., obj_index:next_xy_index])

            output = output.write(
                n,
                K.concatenate(
                    [xy, x[..., wh_index:obj_index], oc],
                    axis=-1,
                ),
            )

        return K.concatenate(
            [output.read(n) for n in range(self.num_masks)], axis=-1
        )
