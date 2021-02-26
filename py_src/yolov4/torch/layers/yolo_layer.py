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
import torch
from torch.nn import Module


class YoloLayer(Module):
    def __init__(self, metalayer, metanet):
        super().__init__()
        self.metalayer = metalayer
        self.metanet = metanet

        num_masks = len(metalayer.mask)
        scale_x_y = metalayer.scale_x_y
        stride = metalayer.classes + 5

        def _coords_0(x):
            """
            @param `x`: Dim(batch, channels, height, width)

            @return: Dim(batch, channels, height, width)
                xy: logistic, scale
                wh: raw
                oc: logistic
            """
            for n in range(num_masks):
                xy_index = n * stride
                wh_index = xy_index + 2
                obj_index = xy_index + 4
                next_xy_index = (n + 1) * stride

                if scale_x_y == 1.0:
                    x[:, xy_index:wh_index] = torch.sigmoid(
                        x[:, xy_index:wh_index]
                    )
                else:
                    x[:, xy_index:wh_index] = scale_x_y * torch.sigmoid(
                        x[:, xy_index:wh_index]
                    ) - (0.5 * (scale_x_y - 1))

                x[:, obj_index:next_xy_index] = torch.sigmoid(
                    x[:, obj_index:next_xy_index]
                )

            return x

        def _coords_1(x):
            """
            @param `x`: Dim(batch, channels, height, width)

            @return: Dim(batch, channels, height, width)
                xy: scale
                wh: raw
                oc: raw
            """
            for n in range(num_masks):
                xy_index = n * stride
                wh_index = xy_index + 2

                x[:, xy_index:wh_index] = scale_x_y * x[
                    :, xy_index:wh_index
                ] - (0.5 * (scale_x_y - 1))

            return x

        if metalayer.new_coords:
            self._yolo_function = _coords_1
        else:
            self._yolo_function = _coords_0

    def forward(self, x):
        """
        @param `x`: Dim(batch, channels, height, width)

        @return: Dim(batch, channels, height, width)
        """
        return self._yolo_function(x)
