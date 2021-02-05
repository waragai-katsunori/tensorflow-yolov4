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


class RouteLayer(BaseLayer):
    def __init__(self, index: int, type_index: int):
        super().__init__(index=index, type_index=type_index, type_name="route")
        self._groups = 1
        self._group_id = 0
        self._layers: tuple

    @property
    def groups(self) -> int:
        return self._groups

    @property
    def group_id(self) -> int:
        return self._group_id

    @property
    def layers(self) -> tuple:
        return self._layers

    def __repr__(self) -> str:
        rep = f"{self._index_:4} route    "
        for layer in self._layers:
            rep += f"{layer:3},"
        rep += "    " * (5 - len(self._layers))
        rep += "                   "
        rep += (
            f"-> {self._output_shape[0]:4} x{self._output_shape[1]:4} x"
            f"{self._output_shape[2]:4}"
        )
        return rep

    def __setitem__(self, key: str, value: Any):
        if key in ("groups", "group_id"):
            self.__setattr__(f"_{key}", int(value))
        elif key in ("layers",):
            self.__setattr__(
                f"_{key}",
                tuple(
                    int(i) if int(i) >= 0 else self._index_ + int(i)
                    for i in value.split(",")
                ),
            )
        elif key == "input_shape":
            self.__setattr__(f"_{key}", value)
            if self._groups != 1:
                # split
                self._output_shape = (
                    self._input_shape[0],
                    self._input_shape[1],
                    self._input_shape[2] // self._groups,
                )
            else:
                if len(self._layers) == 1:
                    # route
                    self._output_shape = self._input_shape
                else:
                    # concatenate
                    output_shape = [*self._input_shape[0]]
                    output_shape[-1] = 0
                    for input_shape in self._input_shape:
                        output_shape[-1] += input_shape[-1]
                    self._output_shape = tuple(output_shape)

        else:
            raise KeyError(f"'{key}' is not supported")
