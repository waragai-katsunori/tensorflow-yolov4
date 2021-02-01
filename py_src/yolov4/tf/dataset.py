"""
MIT License

Copyright (c) 2019 YangYun
Copyright (c) 2020 Việt Hùng
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
import random
from typing import List

import cv2
import numpy as np
from tensorflow import keras

from .train import bbox_iou
from ..common import media
from ..common.config import YOLOConfig
from ..common.parser import parse_dataset


class YOLODataset(keras.utils.Sequence):
    def __init__(
        self,
        config: YOLOConfig,
        dataset_path: str,
        dataset_type: str = "converted_coco",
        image_path_prefix: str = "",
        training: bool = False,
    ):
        self._config = config
        self._dataset = parse_dataset(
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            image_path_prefix=image_path_prefix,
        )
        # Etc ##################################################################

        self._num_yolo = config.count["yolo"]

        self._anchors = []
        for i in range(self._num_yolo):
            name = f"yolo{i}"
            self._anchors.append([])
            for mask in config[name]["mask"]:
                aw, ah = config[f"yolo{i}"]["anchors"][mask]
                self._anchors[i].append(
                    (aw / config["net"]["width"], ah / config["net"]["width"])
                )

        # Data augmentation ####################################################

        self._augmentation: List[str] = []
        if config["net"]["mosaic"]:
            self._augmentation.append("mosaic")

        self.batch_size = self._config["net"]["batch"]

        if training and len(self._augmentation) > 0:
            self.augmentation_batch_size = int(
                self._config["net"]["batch"] * 0.3
            )
            self._training = True
        else:
            self.augmentation_batch_size = 0
            self._training = False

        # Grid #################################################################

        self._strides = tuple(
            round(
                np.sqrt(
                    config["net"]["height"]
                    * config["net"]["width"]
                    / (output_shape[1] / 3)
                ).item()
            )
            for output_shape in config.output_shape
        )

        self._grid_shapes = tuple(
            (
                1,
                config["net"]["height"] // stride,
                config["net"]["width"] // stride,
                3,
                5 + len(config.names),
            )
            for stride in self._strides
        )

        self._grid_xy_list = [
            np.tile(
                np.reshape(
                    np.stack(
                        np.meshgrid(
                            (np.arange(grid_shape[1]) + 0.5) / grid_shape[1],
                            (np.arange(grid_shape[2]) + 0.5) / grid_shape[2],
                        ),
                        axis=-1,
                    ),
                    (1, grid_shape[1], grid_shape[2], 1, 2),
                ),
                (1, 1, 1, 3, 1),
            ).astype(np.float32)
            for grid_shape in self._grid_shapes
        ]

    def _convert_bboxes_to_ground_truth(self, bboxes):
        """
        @param bboxes: [[b_x, b_y, b_w, b_h, class_id], ...]

        @return [yolo0, yolo1, ...]
            Dim(1, grid_y * grid_x * anchors,
                                (b_x, b_y, b_w, b_h, conf, prob_0, prob_1, ...))
        """
        ground_truth = [
            np.zeros(
                grid_shape,
                dtype=np.float32,
            )
            for grid_shape in self._grid_shapes
        ]

        for i in range(self._num_yolo):
            ground_truth[i][..., 0:2] = self._grid_xy_list[i]

        for bbox in bboxes:
            # bbox: [b_x, b_y, b_w, b_h, class_id]
            xywh = np.array(bbox[:4], dtype=np.float32)
            class_id = int(bbox[4])

            # prob_0, prob_1, ...
            onehot = np.zeros(len(self._config.names), dtype=np.float32)
            onehot[class_id] = 1.0

            ious = []
            exist_positive = False
            for i in range(self._num_yolo):
                # Dim(anchors, xywh)
                anchors_xywh = np.zeros((3, 4), dtype=np.float32)
                anchors_xywh[:, 0:2] = xywh[0:2]
                anchors_xywh[:, 2:4] = self._anchors[i]
                iou = bbox_iou(xywh, anchors_xywh)
                ious.append(iou)
                # iou threshold
                iou_mask = iou > 0.3

                if np.any(iou_mask):
                    xy_index = xywh[0:2] * (
                        self._grid_shapes[i][1],  # width
                        self._grid_shapes[i][0],  # height
                    )

                    exist_positive = True
                    for j, mask in enumerate(iou_mask):
                        if mask:
                            _x, _y = int(xy_index[0]), int(xy_index[1])
                            ground_truth[i][0, _y, _x, j, 0:4] = xywh
                            ground_truth[i][0, _y, _x, j, 4:5] = 1.0
                            ground_truth[i][0, _y, _x, j, 5:] = onehot

            if not exist_positive:
                index = np.argmax(np.array(ious))
                i = index // 3
                j = index % 3

                xy_index = xywh[0:2] * (
                    self._grid_shapes[i][1],  # width
                    self._grid_shapes[i][0],  # height
                )

                _x, _y = int(xy_index[0]), int(xy_index[1])
                ground_truth[i][0, _y, _x, j, 0:4] = xywh
                ground_truth[i][0, _y, _x, j, 4:5] = 1.0
                ground_truth[i][0, _y, _x, j, 5:] = onehot

        return [np.reshape(gt, (1, -1, gt.shape[-1])) for gt in ground_truth]

    def _convert_dataset_to_image_and_bboxes(self, dataset):
        """
        @param dataset: [image_path, [[x, y, w, h, class_id], ...]]

        @return image, bboxes
            image: 0.0 ~ 1.0, Dim(1, height, width, channels)
        """
        # pylint: disable=bare-except
        try:
            image = cv2.imread(dataset[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            return None, None

        resized_image, resized_bboxes = media.resize_image(
            image,
            target_shape=self._config.input_shape,
            ground_truth=dataset[1],
        )
        resized_image = np.expand_dims(resized_image / 255.0, axis=0)

        return resized_image, resized_bboxes

    def _get_dataset(self, index: int):
        offset = 0
        for offset in range(5):
            image, bboxes = self._convert_dataset_to_image_and_bboxes(
                self._dataset[(index + offset) % len(self._dataset)]
            )
            if image is None:
                offset += 1
            else:
                return image, bboxes

        raise FileNotFoundError("Failed to find images")

    def __getitem__(self, index):
        """
        @return
            images: Dim(batch, height, width, channels)
            ground_truth: Dim(batch, -1, 5 + len(names))
        """

        batch_x = []
        batch_y = [[] for _ in range(self._num_yolo)]

        start_index = index * self.batch_size

        for i in range(self.batch_size - self.augmentation_batch_size):
            image, bboxes = self._get_dataset(start_index + i)

            batch_x.append(image)
            for j, gt in enumerate(
                self._convert_bboxes_to_ground_truth(bboxes)
            ):
                batch_y[j].append(gt)

        for i in range(self.augmentation_batch_size):
            augmentation = self._augmentation[
                random.randrange(0, len(self._augmentation))
            ]

            image = None
            bboxes = None
            if augmentation == "mosaic":
                image, bboxes = mosaic(
                    *[
                        self._get_dataset(
                            start_index + random.randrange(0, self.batch_size)
                        )
                        for _ in range(4)
                    ]
                )

            batch_x.append(image)
            for j, gt in enumerate(
                self._convert_bboxes_to_ground_truth(bboxes)
            ):
                batch_y[j].append(gt)

        return np.concatenate(batch_x, axis=0), [
            np.concatenate(y, axis=0) for y in batch_y
        ]

    def __len__(self):
        return len(self._dataset) // (
            self.batch_size - self.augmentation_batch_size
        )


def cut_out(dataset):
    """
    @parma `dataset`: [image(float), bboxes]
            bboxes = [image_path, [[x, y, w, h, class_id], ...]]
    """
    _size = dataset[0].shape[1:3]  # height, width
    for bbox in dataset[1]:
        if random.random() < 0.5:
            _pixel_bbox = [
                int(pos * _size[(i + 1) % 2]) for i, pos in enumerate(bbox[0:4])
            ]
            _x_min = _pixel_bbox[0] - (_pixel_bbox[2] // 2)
            _y_min = _pixel_bbox[1] - (_pixel_bbox[3] // 2)
            _cut_out_width = _pixel_bbox[2] // 4
            _cut_out_height = _pixel_bbox[3] // 4
            _x_offset = (
                int((_pixel_bbox[2] - _cut_out_width) * random.random())
                + _x_min
            )
            _y_offset = (
                int((_pixel_bbox[3] - _cut_out_height) * random.random())
                + _y_min
            )
            dataset[0][
                :,
                _y_offset : _y_offset + _cut_out_height,
                _x_offset : _x_offset + _cut_out_width,
                :,
            ] = 0.5

    return dataset


def mix_up(dataset0, dataset1, alpha=0.2):
    return (
        (dataset0[0] * alpha + dataset1[0] * (1 - alpha)),
        (np.concatenate((dataset0[1], dataset1[1]), axis=0)),
    )


def mosaic(dataset0, dataset1, dataset2, dataset3):
    size = dataset0[0].shape[1:3]  # height, width
    image = np.empty((1, size[0], size[1], 3))
    bboxes = []

    partition_x = int((random.random() * 0.6 + 0.2) * size[1])
    partition_y = int((random.random() * 0.6 + 0.2) * size[0])

    x_offset = [0, partition_x, 0, partition_x]
    y_offset = [0, 0, partition_y, partition_y]

    left = [
        (size[1] - partition_x) // 2,
        partition_x // 2,
        (size[1] - partition_x) // 2,
        partition_x // 2,
    ]
    right = [
        left[0] + partition_x,
        left[1] + size[1] - partition_x,
        left[2] + partition_x,
        left[3] + size[1] - partition_x,
    ]
    top = [
        (size[0] - partition_y) // 2,
        (size[0] - partition_y) // 2,
        partition_y // 2,
        partition_y // 2,
    ]
    down = [
        top[0] + partition_y,
        top[1] + partition_y,
        top[2] + size[0] - partition_y,
        top[3] + size[0] - partition_y,
    ]

    image[:, :partition_y, :partition_x, :] = dataset0[0][
        :,
        top[0] : down[0],
        left[0] : right[0],
        :,
    ]
    image[:, :partition_y, partition_x:, :] = dataset1[0][
        :,
        top[1] : down[1],
        left[1] : right[1],
        :,
    ]
    image[:, partition_y:, :partition_x, :] = dataset2[0][
        :,
        top[2] : down[2],
        left[2] : right[2],
        :,
    ]
    image[:, partition_y:, partition_x:, :] = dataset3[0][
        :,
        top[3] : down[3],
        left[3] : right[3],
        :,
    ]

    for i, _bboxes in enumerate(
        (dataset0[1], dataset1[1], dataset2[1], dataset3[1])
    ):
        for bbox in _bboxes:
            pixel_bbox = [
                int(pos * size[(i + 1) % 2]) for i, pos in enumerate(bbox[0:4])
            ]
            x_min = int(pixel_bbox[0] - pixel_bbox[2] // 2)
            y_min = int(pixel_bbox[1] - pixel_bbox[3] // 2)
            x_max = int(pixel_bbox[0] + pixel_bbox[2] // 2)
            y_max = int(pixel_bbox[1] + pixel_bbox[3] // 2)

            class_id = bbox[4]

            if x_min > right[i]:
                continue
            if y_min > down[i]:
                continue
            if x_max < left[i]:
                continue
            if y_max < top[i]:
                continue

            if x_max > right[i]:
                x_max = right[i]
            if y_max > down[i]:
                y_max = down[i]
            if x_min < left[i]:
                x_min = left[i]
            if y_min < top[i]:
                y_min = top[i]

            x_min -= left[i]
            x_max -= left[i]
            y_min -= top[i]
            y_max -= top[i]

            if x_min + 3 > x_max:
                continue

            if y_min + 3 > y_max:
                continue

            bboxes.append(
                np.array(
                    [
                        [
                            ((x_min + x_max) / 2 + x_offset[i]) / size[1],
                            ((y_min + y_max) / 2 + y_offset[i]) / size[0],
                            (x_max - x_min) / size[1],
                            (y_max - y_min) / size[0],
                            class_id,
                        ],
                    ]
                )
            )

    if len(bboxes) == 0:
        return dataset0

    return image, np.concatenate(bboxes, axis=0)
