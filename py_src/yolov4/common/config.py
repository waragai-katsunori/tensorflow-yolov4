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
from typing import Any, Dict, ItemsView, KeysView, List, Tuple, ValuesView

from . import parser


class YOLOConfig:
    def __init__(self):
        self._cfg: Dict[str, Dict[str, Any]] = {}
        self._count: Dict[str, int]
        self._names: Dict[int, str] = {}

        self.output_shape: List[tuple]

    # Parse ####################################################################

    def parse_cfg(self, cfg_path: str):
        self._cfg, self._count = parser.parse_cfg(cfg_path=cfg_path)
        if len(self._names) != 0:
            if self._cfg["yolo0"]["classes"] != len(self._names):
                raise RuntimeError(
                    "YOLOConfig: '[yolo] classes' of 'cfg' and the number of"
                    " 'names' do not match."
                )

    def parse_names(self, names_path: str):
        self._names = parser.parse_names(names_path=names_path)
        if len(self._cfg) != 0:
            if self._cfg["yolo0"]["classes"] != len(self._names):
                raise RuntimeError(
                    "YOLOConfig: '[yolo] classes' of 'cfg' and the number of"
                    " 'names' do not match."
                )

    # Property #################################################################

    @property
    def count(self) -> Dict[str, int]:
        """
        key: layer_type
        value: the number of layers of the same type
        """
        return self._count

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (
            self["net"]["height"],
            self["net"]["width"],
            self["net"]["channels"],
        )

    @property
    def names(self) -> Dict[int, str]:
        return self._names

    @property
    def tiny(self) -> bool:
        return len(self._cfg) < 70

    # Dict #####################################################################

    def items(self) -> ItemsView[str, Any]:
        return self._cfg.items()

    def keys(self) -> KeysView[str]:
        return self._cfg.keys()

    def values(self) -> ValuesView[Any]:
        return self._cfg.values()

    # Magic ####################################################################

    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]
