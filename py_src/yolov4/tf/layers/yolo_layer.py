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

        stride = metalayer.classes + 5
        num_masks = len(metalayer.mask)

        @tf.function
        def _coords_0(x, training):
            """
            @param `x`: Dim(height, width, height, channels)

            @return: Dim(height, width, height, channels)
                xy: logistic, scale
                wh: raw or exp(training)
                oc: logistic
            """
            output = []
            scale_x_y = tf.constant(metalayer.scale_x_y, x.dtype)

            # for n in tf.range -> tf.while_loop
            # but failed to convert to tflite.
            for n in range(num_masks):
                xy_index = n * stride
                wh_index = xy_index + 2
                obj_index = xy_index + 4
                next_xy_index = (n + 1) * stride

                if scale_x_y == 1.0:
                    xy = K.sigmoid(x[..., xy_index:wh_index])
                else:
                    xy = scale_x_y * K.sigmoid(x[..., xy_index:wh_index]) - (
                        0.5 * (scale_x_y - 1)
                    )

                if training:
                    wh = K.exp(x[..., wh_index:obj_index])
                else:
                    wh = x[..., wh_index:obj_index]

                oc = K.sigmoid(x[..., obj_index:next_xy_index])

                output.append(
                    K.concatenate(
                        [xy, wh, oc],
                        axis=-1,
                    )
                )

            return K.concatenate(output, axis=-1)

        @tf.function
        def _coords_1(x, training):
            """
            @param `x`: Dim(height, width, height, channels)

            @return: Dim(height, width, height, channels)
                xy: scale
                wh: raw or pow(training)
                oc: raw
            """
            output = []
            scale_x_y = tf.constant(metalayer.scale_x_y, x.dtype)

            # for n in tf.range -> tf.while_loop
            # but failed to convert to tflite.
            for n in range(num_masks):
                xy_index = n * stride
                wh_index = xy_index + 2
                obj_index = xy_index + 4
                next_xy_index = (n + 1) * stride

                xy = scale_x_y * x[..., xy_index:wh_index] - (
                    0.5 * (scale_x_y - 1)
                )
                if training:
                    wh = K.pow(x[..., wh_index:obj_index] * 2, 2)
                else:
                    wh = x[..., wh_index:obj_index]

                output.append(
                    K.concatenate(
                        [
                            xy,
                            wh,
                            x[..., obj_index:next_xy_index],
                        ],
                        axis=-1,
                    )
                )

            return K.concatenate(output, axis=-1)

        if metalayer.new_coords:
            self._yolo_function = _coords_1
        else:
            self._yolo_function = _coords_0

    def call(self, x, training=False):
        """
        @param `x`: Dim(height, width, height, channels)

        @return: Dim(height, width, height, channels)
        """
        return self._yolo_function(x, training)
