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
from typing import Any

from .base_layer import BaseLayer


class ConvolutionalLayer(BaseLayer):
    def __init__(self, index: int, type_index: int):
        super().__init__(
            index=index, type_index=type_index, type_name="convolutional"
        )
        self._activation = "logistic"
        self._batch_normalize = 0
        self._filters = 1
        self._pad = False
        self._padding = 0
        self._size = 1
        self._stride = 1

    @property
    def activation(self) -> str:
        return self._activation

    @property
    def batch_normalize(self) -> int:
        return self._batch_normalize

    @property
    def filters(self) -> int:
        return self._filters

    @property
    def pad(self) -> bool:
        return self._pad

    @property
    def padding(self) -> int:
        if self._pad:
            return self._size // 2
        return self._padding

    @property
    def size(self) -> int:
        return self._size

    @property
    def stride(self) -> int:
        return self._stride

    def __repr__(self) -> str:
        return (
            f"{self._index_:4} conv    {self.filters:4}      {self._size:2} x"
            f"{self._size:2} / {self._stride}   {self._input_shape[0]:4} x"
            f"{self._input_shape[1]:4} x{self._input_shape[2]:4} "
            f"-> {self._output_shape[0]:4} x{self._output_shape[1]:4} x"
            f"{self._output_shape[2]:4}"
        )

    def __setitem__(self, key: str, value: Any):
        if key in ("activation",):
            self.__setattr__(f"_{key}", str(value))
        elif key in ("batch_normalize", "filters", "padding", "size", "stride"):
            self.__setattr__(f"_{key}", int(value))
        elif key in ("pad",):
            self.__setattr__(f"_{key}", bool(int(value)))
        elif key == "input_shape":
            self.__setattr__(f"_{key}", value)
            self._output_shape = (
                self._input_shape[0] // self._stride,
                self._input_shape[1] // self._stride,
                self._filters,
            )
        else:
            raise KeyError(f"'{key}' is not supported")
