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


class NetLayer(BaseLayer):
    def __init__(self, index: int, type_index: int):
        super().__init__(index=index, type_index=type_index, type_name="net")
        self._batch = 1
        self._burn_in = 0
        self._channels = 0
        self._height = 0
        self._learning_rate = 0.001
        self._max_batches = 0
        self._mosaic = False
        self._policy = "steps"
        self._power = 4
        self._scales: tuple
        self._steps: tuple
        self._width = 0

    @property
    def batch(self) -> int:
        return self._batch

    @property
    def burn_in(self) -> int:
        return self._burn_in

    @property
    def channels(self) -> int:
        # override
        return self._channels

    @property
    def height(self) -> int:
        # override
        return self._height

    @property
    def input_shape(self) -> tuple:
        # override
        return (self._height, self._width, self._channels)

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def max_batches(self) -> int:
        return self._max_batches

    @property
    def mosaic(self) -> bool:
        return self._mosaic

    @property
    def name(self) -> str:
        # override
        return self._type_

    @property
    def output_shape(self) -> tuple:
        # override
        return (self._height, self._width, self._channels)

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def power(self) -> int:
        return self._power

    @property
    def scales(self) -> tuple:
        return self._scales

    @property
    def steps(self) -> tuple:
        return self._steps

    @property
    def width(self) -> int:
        # override
        return self._width

    def __repr__(self) -> str:
        rep = f"batch: {self._batch}"
        return rep

    def __setitem__(self, key: str, value: Any):
        if key in ("policy",):
            self.__setattr__(f"_{key}", str(value))
        elif key in (
            "batch",
            "burn_in",
            "channels",
            "height",
            "max_batches",
            "power",
            "width",
        ):
            self.__setattr__(f"_{key}", int(value))
        elif key in ("mosaic",):
            self.__setattr__(f"_{key}", bool(int(value)))
        elif key in ("learning_rate",):
            self.__setattr__(f"_{key}", float(value))
        elif key in ("steps",):
            self.__setattr__(
                f"_{key}", tuple(int(i.strip()) for i in value.split(","))
            )
        elif key in ("scales",):
            self.__setattr__(
                f"_{key}", tuple(float(i.strip()) for i in value.split(","))
            )
        else:
            raise KeyError(f"'{key}' is not supported")
