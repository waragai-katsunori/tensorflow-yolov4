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
import numpy as np
import torch


def load_weights(model, weights_file: str):
    with open(weights_file, "rb") as fd:
        # major, minor, revision, seen, _
        _np_fromfile(fd, dtype=np.int32, count=5)

        for layer in model.layers:
            if "convolutional" in layer.metalayer.name:
                if not yolo_conv2d_load_weights(layer, fd):
                    break

        if len(fd.read()) != 0:
            raise ValueError("Model and weights file do not match.")


def _np_fromfile(fd, dtype, count: int):
    data = np.fromfile(fd, dtype=dtype, count=count)
    if len(data) != count:
        if len(data) == 0:
            return None
        raise ValueError("Model and weights file do not match.")
    return data


def yolo_conv2d_load_weights(yolo_conv2d, fd) -> bool:
    conv2d = yolo_conv2d[0]
    batch_normalization = None
    if yolo_conv2d.metalayer.batch_normalize:
        batch_normalization = yolo_conv2d[1]

    filters = yolo_conv2d.metalayer.filters

    if batch_normalization is not None:
        # darknet weights: [beta, gamma, mean, variance]
        bn_weights = _np_fromfile(fd, dtype=np.float32, count=4 * filters)
        if bn_weights is None:
            return False
        bn_weights = bn_weights.reshape((4, filters))

        batch_normalization.bias.data.copy_(torch.from_numpy(bn_weights[0]))
        batch_normalization.weight.data.copy_(torch.from_numpy(bn_weights[1]))
        batch_normalization.running_mean.copy_(torch.from_numpy(bn_weights[2]))
        batch_normalization.running_var.copy_(torch.from_numpy(bn_weights[3]))

    conv_bias = None
    if not yolo_conv2d.metalayer.batch_normalize:
        conv_bias = _np_fromfile(fd, dtype=np.float32, count=filters)
        if conv_bias is None:
            return False

    # darknet shape (out_dim, in_dim, kernel_size, kernel_size)
    conv_shape = (
        filters,
        yolo_conv2d.metalayer.channels,
        yolo_conv2d.metalayer.size,
        yolo_conv2d.metalayer.size,
    )

    conv_weights = _np_fromfile(
        fd, dtype=np.float32, count=np.product(conv_shape)
    )
    if conv_weights is None:
        return False

    conv2d.weight.data.copy_(
        torch.from_numpy(conv_weights).reshape(conv2d.weight.data.shape)
    )

    if conv_bias is not None:
        conv2d.bias.data.copy_(torch.from_numpy(conv_bias))

    return True


def save_weights(model, weights_file: str, to: str = ""):
    with open(weights_file, "wb") as fd:
        # major, minor, revision, seen, _
        np.array([0, 2, 5, 32032000, 0], dtype=np.int32).tofile(fd)

        for layer in model.layers:
            if "convolutional" in layer.metalayer.name:
                yolo_conv2d_save_weights(layer, fd)
                if layer.metalayer.name == to:
                    break


def yolo_conv2d_save_weights(yolo_conv2d, fd):
    conv2d = yolo_conv2d[0]
    batch_normalization = None
    if yolo_conv2d.metalayer.batch_normalize:
        batch_normalization = yolo_conv2d[1]

    if batch_normalization is not None:
        batch_normalization.bias.data.numpy().tofile(fd)
        batch_normalization.weight.data.numpy().tofile(fd)
        batch_normalization.running_mean.numpy().tofile(fd)
        batch_normalization.running_var.numpy().tofile(fd)

    if not yolo_conv2d.metalayer.batch_normalize:
        conv2d.bias.data.numpy().tofile(fd)
        conv2d.weight.data.numpy().tofile(fd)
    else:
        conv2d.weight.data.numpy().tofile(fd)
