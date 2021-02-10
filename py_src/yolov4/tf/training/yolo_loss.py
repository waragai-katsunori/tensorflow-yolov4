"""
MIT License

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
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss

from .iou import bbox_iou, bbox_ciou, bbox_giou
from ..model import YOLOv4Model
from ...common.config import YOLOConfig

_BBOX_XIOU_MAP = {
    "iou": bbox_iou,
    "ciou": bbox_ciou,
    "giou": bbox_giou,
}


class YOLOv4Loss(Loss):
    def __init__(self, config: YOLOConfig, model: YOLOv4Model):
        super().__init__(name="YOLOv4Loss")
        self._metayolos = []
        for i in range(config.layer_count["yolo"]):
            self._metayolos.append(config.find_metalayer("yolo", i))
        self._metanet = config.net
        self.model = model

        self._bbox_xiou = _BBOX_XIOU_MAP[self._metayolos[-1].iou_loss]
        self._num_mask = len(self._metayolos[-1].mask)
        self._stride = self._metayolos[-1].classes + 5

    def call(self, y_true, y_pred):
        """
        @param `y_true`: Dim(batch, yolo.h, yolo.w, yolo.c)
        @param `y_pred`: Dim(batch, yolo.h, yolo.w, yolo.c + len(mask))
        """
        yolo_name = y_pred.name.split("/")[-2]
        yolo_index = int(yolo_name.split("_")[-1])
        metayolo = self._metayolos[yolo_index]

        cls_normalizer = metayolo.cls_normalizer
        iou_normalizer = metayolo.iou_normalizer
        obj_normalizer = metayolo.obj_normalizer
        one_index = y_pred.shape[-1]

        def anchor_loop(anchor, iou_loss0, obj_loss0, cls_loss0):
            box_index = anchor * self._stride
            obj_index = box_index + 4
            cls_index = box_index + 5
            next_box_index = box_index + self._stride

            true_box = y_true[..., box_index:obj_index]
            true_obj = y_true[..., obj_index]
            true_cls = y_true[..., cls_index:next_box_index]
            true_one = y_true[..., one_index + anchor]

            pred_box = y_pred[..., box_index:obj_index]
            pred_obj = y_pred[..., obj_index]
            pred_cls = y_pred[..., cls_index:next_box_index]

            # obj loss
            obj_loss1 = obj_normalizer * K.sum(
                K.binary_crossentropy(true_obj, pred_obj)
            )

            # cls loss
            cls_loss1 = cls_normalizer * K.sum(
                true_one
                * K.sum(
                    K.binary_crossentropy(true_cls, pred_cls),
                    axis=-1,
                )
            )

            xious, ious = self._bbox_xiou(pred_box, true_box)

            # xiou loss
            iou_loss1 = iou_normalizer * K.sum(true_one * (1 - xious))

            # metrics update
            ious = true_one * ious
            self.model._ious.assign_add(K.sum(ious))
            self.model._recall50.assign_add(
                K.sum(tf.cast(ious > 0.5, dtype=tf.int32))
            )
            self.model._recall75.assign_add(
                K.sum(tf.cast(ious > 0.75, dtype=tf.int32))
            )

            return (
                tf.add(anchor, 1),
                tf.add(iou_loss0, iou_loss1),
                tf.add(obj_loss0, obj_loss1),
                tf.add(cls_loss0, cls_loss1),
            )

        _, iou_loss0, obj_loss0, cls_loss0 = tf.while_loop(
            cond=lambda anchor, _, __, ___: tf.less(anchor, self._num_mask),
            body=anchor_loop,
            loop_vars=[0, 0.0, 0.0, 0.0],
        )

        self.model._total_truth.assign_add(
            tf.cast(K.sum(y_true[..., one_index:]), dtype=tf.int64)
        )

        iou_loss0 /= self._metanet.batch
        obj_loss0 /= self._metanet.batch
        cls_loss0 /= self._metanet.batch
        total_loss = iou_loss0 + obj_loss0 + cls_loss0

        self.model._iou_loss.assign_add(iou_loss0)
        self.model._obj_loss.assign_add(obj_loss0)
        self.model._cls_loss.assign_add(cls_loss0)
        self.model._total_loss.assign_add(total_loss)

        return total_loss
