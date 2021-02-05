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
from typing import Any, Dict, Tuple

from . import parser


class YOLOConfig:
    def __init__(self):
        self._metalayers: Dict[str, Any] = {}
        self._layer_count: Dict[str, int]
        self._model_name: str
        self._names: Dict[int, str] = {}

        self.output_shape: Tuple[tuple]
        self.with_head: bool

    def find_metalayer(self, layer_type: str, layer_index: int) -> Any:
        """
        Usage:
            last_yolo_layer = config.find_metalayer("yolo", -1)
        """
        if layer_index < 0:
            count = self._layer_count[layer_type]
            layer_index = count + layer_index

        return self._metalayers[f"{layer_type}_{layer_index}"]

    # Parse ####################################################################

    def parse_cfg(self, cfg_path: str):
        (
            self._metalayers,
            self._layer_count,
            self._model_name,
        ) = parser.parse_cfg(cfg_path=cfg_path)
        if len(self._names) != 0:
            yolo_0 = self._metalayers["yolo_0"]
            if yolo_0.classes != len(self._names):
                raise RuntimeError(
                    "YOLOConfig: '[yolo] classes' of 'cfg' and the number of"
                    " 'names' do not match."
                )

    def parse_names(self, names_path: str):
        self._names = parser.parse_names(names_path=names_path)
        if len(self._metalayers) != 0:
            yolo_0 = self._metalayers["yolo_0"]
            if yolo_0.classes != len(self._names):
                raise RuntimeError(
                    "YOLOConfig: '[yolo] classes' of 'cfg' and the number of"
                    " 'names' do not match."
                )

    # Property #################################################################

    @property
    def layer_count(self) -> Dict[str, int]:
        """
        key: layer_type
        value: the number of layers of the same type
        """
        return self._layer_count

    @property
    def metalayers(self) -> Dict[str, Any]:
        return self._metalayers

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def names(self) -> Dict[int, str]:
        """
        class names
        """
        return self._names

    # Magic ####################################################################

    def __getattr__(self, metalayer: str) -> Any:
        return self._metalayers[metalayer]
