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


class YoloLayer(BaseLayer):
    def __init__(self, index: int, type_index: int):
        super().__init__(index=index, type_index=type_index, type_name="yolo")
        self._anchors: tuple
        self._classes = 20
        self._cls_normalizer = 1.0
        self._ignore_thresh = 0.5
        self._iou_loss = "iou"
        self._iou_thresh = 1.0
        self._iou_thresh_kind = "iou"
        self._iou_normalizer = 0.75
        self._label_smooth_eps = 0.0
        self._mask: tuple
        self._max = 200
        self._num = 1
        self._obj_normalizer = 1.0
        self._scale_x_y = 1.0

    @property
    def anchors(self) -> tuple:
        return self._anchors

    @property
    def bflops(self) -> float:
        return 0

    @property
    def classes(self) -> int:
        return self._classes

    @property
    def cls_normalizer(self) -> float:
        return self._cls_normalizer

    @property
    def ignore_thresh(self) -> float:
        return self._ignore_thresh

    @property
    def iou_loss(self) -> str:
        return self._iou_loss

    @property
    def iou_thresh(self) -> float:
        """
        Recommended to use iou_thresh=0.213
        """
        return self._iou_thresh

    @property
    def iou_thresh_kind(self) -> str:
        return self._iou_thresh_kind

    @property
    def iou_normalizer(self) -> float:
        return self._iou_normalizer

    @property
    def label_smooth_eps(self) -> float:
        return self._label_smooth_eps

    @property
    def mask(self) -> tuple:
        return self._mask

    @property
    def max(self) -> int:
        return self._max

    @property
    def obj_normalizer(self) -> float:
        return self._obj_normalizer

    @property
    def scale_x_y(self) -> float:
        return self._scale_x_y

    @property
    def total(self) -> int:
        return self._num

    def __repr__(self) -> str:
        rep = f"{self.index:4}  "
        rep += f"{self.type_name}__"
        rep += f"{self.type_index:<3}  "
        rep += f"iou_loss: {self._iou_loss}, "
        rep += f"iou_norm: {self._iou_normalizer}, "
        rep += f"obj_norm: {self._obj_normalizer}, "
        rep += f"cls_norm: {self._cls_normalizer}, "
        rep += "\n                 "
        rep += f"scale_x_y: {self._scale_x_y}"
        return rep

    def __setitem__(self, key: str, value: Any):
        if key in (
            "iou_loss",
            "iou_thresh_kind",
        ):
            self.__setattr__(f"_{key}", str(value))
        elif key in ("classes", "max", "num"):
            self.__setattr__(f"_{key}", int(value))
        elif key in (
            "cls_normalizer",
            "ignore_thresh",
            "iou_thresh",
            "iou_normalizer",
            "label_smooth_eps",
            "obj_normalizer",
            "scale_x_y",
        ):
            self.__setattr__(f"_{key}", float(value))
        elif key in ("mask",):
            self.__setattr__(
                f"_{key}", tuple(int(i.strip()) for i in value.split(","))
            )
        elif key == "anchors":
            value = [int(i.strip()) for i in value.split(",")]
            _value = []
            for i in range(len(value) // 2):
                _value.append((value[2 * i], value[2 * i + 1]))
            self.__setattr__(f"_{key}", tuple(_value))
        elif key == "input_shape":
            self.__setattr__(f"_{key}", value)
            self._output_shape = self._input_shape
        else:
            raise KeyError(f"'{key}' is not supported")
