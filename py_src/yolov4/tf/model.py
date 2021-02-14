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
from tensorflow.keras import Model

from .layers import get_layer_from_metalayer
from ..common.config import YOLOConfig


class YOLOv4Model(Model):
    def __init__(self, config: YOLOConfig):
        super().__init__(name=config.model_name)

        self._model_layers = []
        for i in range(config.layer_count["total"]):
            metalayer = config.metalayers[i]
            self._model_layers.append(
                get_layer_from_metalayer(
                    metalayer=metalayer, metanet=config.net
                )
            )

    def call(self, x):
        output = []
        return_val = []

        for i, layer in enumerate(self._model_layers):

            if layer.metalayer.type == "route":
                if layer.metalayer.groups != 1:
                    i = layer.metalayer.layers[0]
                    output.append(layer(output[i]))
                else:
                    if len(layer.metalayer.layers) == 1:
                        i = layer.metalayer.layers[0]
                        output.append(layer(output[i]))
                    else:
                        output.append(
                            layer(
                                [output[i] for i in layer.metalayer.layers],
                            )
                        )

            elif layer.metalayer.type == "shortcut":
                # from -> layers
                output.append(
                    layer(
                        [output[i] for i in layer.metalayer.layers],
                    )
                )

            else:
                if i == 0:
                    output.append(layer(x))
                else:
                    output.append(layer(output[i - 1]))

                if layer.metalayer.type in ("yolo", "yolo_tpu"):
                    return_val.append(output[i])

        return return_val
