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
from collections import OrderedDict

import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    LeakyReLU,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
import torch.nn.functional as F


class ConvolutionalLayer(Sequential):
    def __init__(self, metalayer, metanet):
        self.metalayer = metalayer
        self.metanet = metanet

        modules = OrderedDict()

        modules[metalayer.name] = Conv2d(
            in_channels=metalayer.channels,
            out_channels=metalayer.filters,
            kernel_size=metalayer.size,
            stride=metalayer.stride,
            padding=metalayer.padding,
            bias=not metalayer.batch_normalize,
        )

        if metalayer.batch_normalize:
            modules[
                "batch_normalization_" + str(metalayer.index)
            ] = BatchNorm2d(
                num_features=metalayer.filters, momentum=metanet.momentum
            )

        if metalayer.activation == "leaky":
            modules["leaky_" + str(metalayer.index)] = LeakyReLU(
                negative_slope=0.1
            )
        elif metalayer.activation == "linear":
            pass
        elif metalayer.activation == "logistic":
            modules["logistic_" + str(metalayer.index)] = Sigmoid()
        elif metalayer.activation == "mish":
            modules["mish_" + str(metalayer.index)] = Mish()
        elif metalayer.activation == "relu":
            modules["relu_" + str(metalayer.index)] = ReLU()
        else:
            raise ValueError(
                f"YOLOConv2D: '{metalayer.activation}' is not supported."
            )

        super().__init__(modules)


"""
digantamisra98/Mish/Mish/Torch/mish.py

MIT License

Copyright (c) 2019 Diganta Misra
"""


class Mish(Module):
    def forward(self, x):
        """
        Forward pass of the function.
        """
        return x * torch.tanh(F.softplus(x))
