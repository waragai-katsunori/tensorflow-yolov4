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


class BaseLayer:
    def __init__(self, index: int, type_index: int, type_name: str):
        self._index_ = index
        self._input_shape: tuple
        self._output_shape: tuple
        self._type_index_ = type_index
        self._type_name_ = type_name

    @property
    def index(self) -> int:
        return self._index_

    @property
    def input_shape(self) -> tuple:
        return tuple(self._input_shape)

    @property
    def name(self) -> str:
        return f"{self._type_name_}_{self._type_index_}"

    @property
    def output_shape(self) -> tuple:
        return tuple(self._output_shape)

    @property
    def type_index(self) -> int:
        return self._type_index_

    @property
    def type_name(self) -> str:
        return self._type_name_
