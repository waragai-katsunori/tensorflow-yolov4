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
from typing import List

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence

from .augmentation import mosaic
from ..training.iou import bbox_iou
from ...common import media
from ...common.config import YOLOConfig
from ...common.parser import parse_dataset


class YOLODataset(Sequence):
    def __init__(
        self,
        config: YOLOConfig,
        dataset_path: str,
        dataset_type: str = "converted_coco",
        image_path_prefix: str = "",
        training: bool = True,
    ):
        self.dataset = parse_dataset(
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            image_path_prefix=image_path_prefix,
        )
        self._metayolos = []
        for i in range(config.layer_count["yolo"]):
            self._metayolos.append(config.find_metalayer("yolo", i))
        self._metanet = config.net

        _anchors = []
        for anchor in self._metayolos[-1].anchors:
            _anchors.append(
                [
                    0,
                    0,
                    anchor[0] / self._metanet.width,
                    anchor[1] / self._metanet.height,
                ]
            )
        self._anchors = tf.convert_to_tensor(_anchors)

        # Data augmentation ####################################################

        self._augmentation: List[str] = []
        if config.net.mosaic:
            self._augmentation.append("mosaic")

        if training and len(self._augmentation) > 0:
            self._augmentation_batch = int(config.net.batch * 0.3)
            self._training = True
        else:
            self._augmentation_batch = 0
            self._training = False

    def _convert_bboxes_to_ground_truth(self, bboxes):
        """
        @param `bboxes`: [[b_x, b_y, b_w, b_h, class_id], ...]

        @return `groud_truth_one`:
            [Dim(yolo.h, yolo.w, yolo.c + len(mask))] * len(yolo)
        """
        y_true = []
        nums = []
        label_true = 1
        for metayolo in self._metayolos:
            lh, lw, lc = metayolo.output_shape
            # x, y, w, h, o, c0, c1, ..., one, one, ...
            gt_one = np.zeros((lh, lw, lc + len(metayolo.mask)))

            stride = 5 + metayolo.classes
            for n in range(len(metayolo.mask)):
                box_index = n * stride
                cls_index = box_index + 5
                next_box_index = box_index + stride

                if metayolo.label_smooth_eps > 0.0:
                    label_true = 1 - 0.5 * metayolo.label_smooth_eps
                    gt_one[..., cls_index:next_box_index] = (
                        0.5 * metayolo.label_smooth_eps
                    )

            nums.append(np.full((lh, lw, len(metayolo.mask)), np.inf))
            y_true.append(gt_one)

        for t in range(min(self._metayolos[-1].max, len(bboxes))):
            truth = bboxes[t][:4]
            cls_id = int(bboxes[t][4])
            truth_shift = tf.convert_to_tensor([0, 0, truth[2], truth[3]])
            ious = []

            # Find best Anchor
            ious, _ = bbox_iou(truth_shift, self._anchors)
            best_n = K.argmax(ious)

            # Find over iou_thresh
            masks = []
            iou_thresh = self._metayolos[-1].iou_thresh
            if iou_thresh < 1.0:
                for n, iou in enumerate(ious):
                    # TODO: box_iou_kind
                    if iou > iou_thresh and n != best_n:
                        masks.append(n)

            masks.append(best_n)

            for metayolo, gt_one, num in zip(self._metayolos, y_true, nums):
                l_h, l_w, l_c = metayolo.output_shape

                i = int(truth[0] * l_w)
                j = int(truth[1] * l_h)
                stride = 5 + metayolo.classes

                # Get mask for metayolo
                y_mask = []
                for mask in masks:
                    if mask in metayolo.mask:
                        y_mask.append(mask)

                for n, mask in enumerate(metayolo.mask):
                    if mask in y_mask:
                        box_index = n * stride
                        obj_index = box_index + 4
                        cls_index = box_index + 5

                        # Accumulate box and obj
                        gt_one[j, i, box_index:obj_index] += truth
                        gt_one[j, i, obj_index] = 1
                        gt_one[j, i, cls_index + cls_id] = label_true
                        if num[j, i, n] > 1e3:
                            num[j, i, n] = 1
                        else:
                            num[j, i, n] += 1

        # Calculate average box
        for metayolo, gt_one, num in zip(self._metayolos, y_true, nums):
            l_c = metayolo.output_shape[-1]
            stride = 5 + metayolo.classes

            for n in range(len(metayolo.mask)):
                box_index = n * stride
                obj_index = box_index + 4
                one_index = l_c + n
                gt_one[..., box_index:obj_index] /= num[..., n : n + 1]
                gt_one[..., one_index] = gt_one[..., obj_index]

        # TODO: test code

        return y_true

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
            target_shape=self._metanet.input_shape,
            ground_truth=dataset[1],
        )
        resized_image = np.expand_dims(resized_image / 255.0, axis=0)

        return resized_image, resized_bboxes

    def _get_dataset(self, index: int):
        offset = 0
        for offset in range(5):
            image, bboxes = self._convert_dataset_to_image_and_bboxes(
                self.dataset[(index + offset) % len(self.dataset)]
            )
            if image is None:
                offset += 1
            else:
                return image, bboxes

        raise FileNotFoundError("Failed to find images")

    def __getitem__(self, index):
        """
        @return
            `images`: Dim(batch, height, width, channels)
            `groud_truth_one`:
                [Dim(batch, yolo.h, yolo.w, yolo.c + len(mask))] * len(yolo)
        """

        batch_x = []
        # [[gt_one, gt_one, ...],
        #  [gt_one, gt_one, ...], ...]
        batch_y = [[] for _ in range(len(self._metayolos))]

        start_index = index * self._metanet.batch

        for i in range(self._metanet.batch - self._augmentation_batch):
            image, bboxes = self._get_dataset(start_index + i)

            batch_x.append(image)
            ground_truth = self._convert_bboxes_to_ground_truth(bboxes)
            for j in range(len(self._metayolos)):
                batch_y[j].append(ground_truth[j])

        for i in range(self._augmentation_batch):
            augmentation = self._augmentation[
                np.random.randint(0, len(self._augmentation))
            ]

            image = None
            bboxes = None
            if augmentation == "mosaic":
                image, bboxes = mosaic(
                    *[
                        self._get_dataset(
                            start_index
                            + np.random.randint(
                                0,
                                self._metanet.batch - self._augmentation_batch,
                            )
                        )
                        for _ in range(4)
                    ]
                )

            ground_truth = self._convert_bboxes_to_ground_truth(bboxes)
            for j in range(len(self._metayolos)):
                batch_y[j].append(ground_truth[j])

        return np.concatenate(batch_x, axis=0), [
            np.stack(y, axis=0) for y in batch_y
        ]

    def __len__(self):
        return len(self.dataset) // (
            self._metanet.batch - self._augmentation_batch
        )
