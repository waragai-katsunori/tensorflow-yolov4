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
from os import path
import pathlib
import random
from typing import Any, Dict, Tuple, Union

import numpy as np

from .metalayer import (
    ConvolutionalLayer,
    MaxpoolLayer,
    NetLayer,
    RouteLayer,
    ShortcutLayer,
    UpsampleLayer,
    YoloLayer,
)


def parse_cfg(
    cfg_path: str,
) -> Tuple[Dict[Union[str, int], Any], Dict[str, int], str]:
    """
    @return
        Dict[layer_name or layer_index, metalayer]
        Dict[layer_type, count]
        model_name
    """
    metalayers: Dict[Union[str, int], Any] = {}
    count: Dict[str, int] = {
        "convolutional": 0,
        "maxpool": 0,
        "net": 0,
        "route": 0,
        "shortcut": 0,
        "total": -1,
        "upsample": 0,
        "yolo": 0,
    }
    layer_type: str = "net"

    meta_layer: Dict[str, Any] = {
        "convolutional": ConvolutionalLayer,
        "maxpool": MaxpoolLayer,
        "net": NetLayer,
        "route": RouteLayer,
        "shortcut": ShortcutLayer,
        "upsample": UpsampleLayer,
        "yolo": YoloLayer,
    }

    with open(cfg_path, "r") as fd:
        layer = NetLayer(index=-1, type_index=-1)
        for line in fd:
            line = line.strip().split("#")[0]
            if line == "":
                continue

            if line[0] == "[":
                layer_type = line[1:-1]
                count["total"] += 1
                count[layer_type] += 1

                layer = meta_layer[layer_type](
                    index=count["total"] - 1, type_index=count[layer_type] - 1
                )
                metalayers[layer.name] = layer
                metalayers[count["total"] - 1] = layer

            else:
                # layer option
                option, value = line.split("=")
                option = option.strip()
                value = value.strip()
                try:
                    metalayers[layer.name][option] = value
                except KeyError as error:
                    raise RuntimeError(
                        f"parse_cfg: [{layer.name}] '{option}' is not"
                        " supported."
                    ) from error

    # Build layer
    for index in range(count["total"]):
        layer = metalayers[index]

        output_shape = metalayers[index - 1].output_shape
        if layer.type in ("route", "shortcut"):
            if len(layer.layers) > 1:
                output_shape = [
                    metalayers[i].output_shape for i in layer.layers
                ]
            else:
                output_shape = metalayers[layer.layers[0]].output_shape
        layer["input_shape"] = output_shape

    model_name = pathlib.Path(cfg_path).stem

    return metalayers, count, model_name


def parse_names(names_path: str) -> Dict[int, str]:
    """
    @return {id: class name}
    """
    names: Dict[int, str] = {}
    with open(names_path, "r") as fd:
        index = 0
        for class_name in fd:
            class_name = class_name.strip()
            if len(class_name) != 0:
                names[index] = class_name
                index += 1

    return names


def parse_dataset(
    dataset_path: str,
    dataset_type: str = "converted_coco",
    image_path_prefix: str = "",
):
    """
    x: center x 0.0 ~ 1.0
    y: center y 0.0 ~ 1.0
    @return [
                [
                    image_path,
                    [
                        [x, y, w, h, class_id]
                        ,
                        ...
                    ]
                ],
                ...
            ]
    """
    dataset = []

    with open(dataset_path, "r") as fd:
        lines = fd.readlines()

        if dataset_type == "converted_coco":
            for line in lines:
                # line: "<image_path> class_id,x,y,w,h ..."
                bboxes = line.strip().split()

                image_path = bboxes[0]
                if image_path_prefix != "":
                    image_path = path.join(image_path_prefix, image_path)

                xywhc_s = np.zeros((len(bboxes) - 1, 5))
                for i, bbox in enumerate(bboxes[1:]):
                    # bbox = class_id,x,y,w,h
                    bbox = list(map(float, bbox.split(",")))
                    xywhc_s[i, :] = (
                        *bbox[1:],
                        bbox[0],
                    )

                dataset.append([image_path, xywhc_s])

        elif dataset_type == "yolo":
            for line in lines:
                # line: "<image_path>"
                image_path = line.strip()
                if image_path_prefix != "":
                    image_path = path.join(image_path_prefix, image_path)

                root, _ = path.splitext(image_path)
                with open(root + ".txt") as fd2:
                    bboxes = fd2.readlines()
                    xywhc_s = np.zeros((len(bboxes), 5))
                    for i, bbox in enumerate(bboxes):
                        # bbox = class_id x y w h
                        bbox = bbox.strip()
                        bbox = list(map(float, bbox.split(" ")))
                        xywhc_s[i, :] = (
                            *bbox[1:],
                            bbox[0],
                        )
                    dataset.append([image_path, xywhc_s])

    if len(dataset) == 0:
        raise RuntimeError(
            f"parse_dataset: There is no dataset in '{dataset_path}'."
        )

    # Select 5 sets randomly and check the data format
    for _ in range(5):
        first_bbox = dataset[random.randint(0, len(dataset) - 1)][1][0]
        for i in range(4):
            if first_bbox[i] < 0 or first_bbox[i] > 1:
                raise RuntimeError(
                    "parse_dataset: 'center_x', 'center_y', 'width', and"
                    " 'height' are between 0.0 and 1.0."
                )

        if int(first_bbox[4]) < 0:
            raise RuntimeError(
                "parse_dataset: 'class_id' is an integer greater than or equal"
                " to 0."
            )

    return dataset
