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
from typing import Any, Callable, Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras

from .layers import YOLOConv2D
from ..common.config import YOLOConfig


class YOLOv4Model(keras.Model):
    def __init__(self, config: YOLOConfig):
        self._model_config: YOLOConfig = config
        super().__init__(name=config.model_name)

        # Model ################################################################

        _l2 = keras.regularizers.L2(l2=0.005)

        self._model_layers = []

        for index in range(config.layer_count["total"]):
            metalayer = config.metalayers[index]

            if metalayer.type_name == "convolutional":
                self._model_layers.append(
                    YOLOConv2D(
                        activation=metalayer.activation,
                        batch_normalize=metalayer.batch_normalize,
                        filters=metalayer.filters,
                        kernel_regularizer=_l2,
                        name=metalayer.name,
                        padding=metalayer.padding,
                        size=metalayer.size,
                        stride=metalayer.stride,
                    )
                )

            elif metalayer.type_name == "route":
                if metalayer.groups != 1:
                    self._model_layers.append(
                        keras.layers.Lambda(
                            _split_and_get(
                                groups=metalayer.groups,
                                group_id=metalayer.group_id,
                            ),
                            name=metalayer.name,
                        )
                    )
                else:
                    if len(metalayer.layers) == 1:
                        self._model_layers.append(
                            keras.layers.Lambda(
                                lambda x: x, name=metalayer.name
                            )
                        )
                    else:
                        self._model_layers.append(
                            keras.layers.Concatenate(
                                axis=-1, name=metalayer.name
                            )
                        )

            elif metalayer.type_name == "shortcut":
                self._model_layers.append(keras.layers.Add(name=metalayer.name))

            elif metalayer.type_name == "maxpool":
                self._model_layers.append(
                    keras.layers.MaxPooling2D(
                        name=metalayer.name,
                        padding="same",
                        pool_size=(metalayer.size, metalayer.size),
                        strides=(
                            metalayer.stride,
                            metalayer.stride,
                        ),
                    )
                )

            elif metalayer.type_name == "upsample":
                self._model_layers.append(
                    keras.layers.UpSampling2D(
                        interpolation="bilinear", name=metalayer.name
                    )
                )

            elif metalayer.type_name == "yolo":
                if config.with_head:
                    self._model_layers.append(
                        YOLOv3Head(config=config, name=metalayer.name)
                    )
                else:
                    self._model_layers.append(
                        keras.layers.Lambda(lambda x: x, name=metalayer.name)
                    )

        # Training #############################################################

        self.training_iterations = 0

    def call(self, x):
        output = []
        return_val = []

        for index in range(self._model_config.layer_count["total"]):
            metalayer = self._model_config.metalayers[index]
            layer_function = self._model_layers[index]

            if metalayer.type_name == "route":
                if metalayer.groups != 1:
                    index = metalayer.layers[0]
                    output.append(layer_function(output[index]))
                else:
                    if len(metalayer.layers) == 1:
                        index = metalayer.layers[0]
                        output.append(layer_function(output[index]))
                    else:
                        output.append(
                            layer_function(
                                [output[i] for i in metalayer.layers],
                            )
                        )

            elif metalayer.type_name == "shortcut":
                # from -> layers
                output.append(
                    layer_function(
                        [output[i] for i in metalayer.layers],
                    )
                )

            else:
                if index == 0:
                    output.append(layer_function(x))
                else:
                    output.append(layer_function(output[index - 1]))

                if metalayer.type_name == "yolo":
                    return_val.append(output[index])

        return return_val


class YOLOv3Head(keras.Model):
    def __init__(self, config: YOLOConfig, name: str):
        super().__init__(name=name)
        metalayer = config.metalayers[self.name]
        self._anchors = tuple(
            metalayer.anchors[mask] for mask in metalayer.mask
        )

        self._grid_coord: Tuple[tuple]

        self._inver_grid_wh: Tuple[float, float]

        self._inver_image_wh = (
            1 / config.net.width,
            1 / config.net.height,
        )

        self._return_shape: Tuple[int, int, int]

        self._scale_x_y = metalayer.scale_x_y

    def build(self, grid_shape):
        _, grid_height, grid_width, filters = grid_shape

        grid_coord = []
        for y in range(grid_height):
            grid_coord.append([])
            for x in range(grid_width):
                grid_coord[y].append((x / grid_width, y / grid_height))
            grid_coord[y] = tuple(grid_coord[y])

        self._grid_coord = tuple(grid_coord)

        self._inver_grid_wh = (1 / grid_width, 1 / grid_height)

        self._return_shape = (-1, grid_height * grid_width, filters // 3)

    def call(self, x):
        raw_split = tf.split(x, 3, axis=-1)

        sig = keras.activations.sigmoid(x)
        sig_split = tf.split(sig, 3, axis=-1)

        output = []
        for i in range(3):

            # Operation not supported on Edge TPU
            xy, _, oc = tf.split(sig_split[i], [2, 2, -1], axis=-1)
            _, wh, _ = tf.split(raw_split[i], [2, 2, -1], axis=-1)
            wh = tf.math.exp(wh)

            # Can be Mapped to Edge TPU
            if self._scale_x_y != 1.0:
                xy = (xy - 0.5) * self._scale_x_y + 0.5
            xy *= self._inver_grid_wh
            xy += self._grid_coord

            wh = wh * self._anchors[i] * self._inver_image_wh

            output.append(
                tf.reshape(tf.concat([xy, wh, oc], axis=-1), self._return_shape)
            )

        return tf.concat(output, axis=1)


def _split_and_get(groups: int, group_id: int) -> Callable:
    return lambda x: tf.split(
        x,
        groups,
        axis=-1,
    )[group_id]
