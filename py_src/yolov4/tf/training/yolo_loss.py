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
        self._num_mask = tf.constant(
            len(self._metayolos[-1].mask), dtype=tf.int32
        )

        # yolo #################################################################

        self._iou_norm = tf.constant(
            [metayolo.iou_normalizer for metayolo in self._metayolos],
            dtype=tf.float32,
        )
        self._cls_norm = tf.constant(
            [metayolo.cls_normalizer for metayolo in self._metayolos],
            dtype=tf.float32,
        )
        self._obj_norm = tf.constant(
            [metayolo.obj_normalizer for metayolo in self._metayolos],
            dtype=tf.float32,
        )

        # anchor ###############################################################

        stride = self._metayolos[-1].classes + 5
        self._box_index = tf.constant(
            [stride * i for i in range(len(self._metayolos[-1].mask))],
            dtype=tf.int32,
        )
        self._obj_index = tf.constant(
            [stride * i + 4 for i in range(len(self._metayolos[-1].mask))],
            dtype=tf.int32,
        )
        self._cls_index = tf.constant(
            [stride * i + 5 for i in range(len(self._metayolos[-1].mask))],
            dtype=tf.int32,
        )
        self._next_box_index = tf.constant(
            [stride * (i + 1) for i in range(len(self._metayolos[-1].mask))],
            dtype=tf.int32,
        )
        self._one_index = tf.constant(
            [
                self._metayolos[-1].channels + i
                for i in range(len(self._metayolos[-1].mask))
            ],
            dtype=tf.int32,
        )

    def call(self, y_true, y_pred):
        """
        @param `y_true`: Dim(batch, yolo.h, yolo.w, yolo.c)
        @param `y_pred`: Dim(batch, yolo.h, yolo.w, yolo.c + len(mask))
        """
        yolo_name = y_pred.name.split("/")[-2]
        yolo_index = int(yolo_name.split("_")[-1])

        def anchor_loop(anchor, iou_loss0, obj_loss0, cls_loss0):
            true_box = y_true[
                ..., self._box_index[anchor] : self._obj_index[anchor]
            ]
            true_obj = y_true[..., self._obj_index[anchor]]
            true_cls = y_true[
                ..., self._cls_index[anchor] : self._next_box_index[anchor]
            ]
            true_one = y_true[..., self._one_index[anchor]]

            pred_box = y_pred[
                ..., self._box_index[anchor] : self._obj_index[anchor]
            ]
            pred_obj = y_pred[..., self._obj_index[anchor]]
            pred_cls = y_pred[
                ..., self._cls_index[anchor] : self._next_box_index[anchor]
            ]

            # obj loss
            obj_loss1 = self._obj_norm[yolo_index] * K.sum(
                K.binary_crossentropy(true_obj, pred_obj)
            )

            # cls loss
            cls_loss1 = self._cls_norm[yolo_index] * K.sum(
                true_one
                * K.sum(
                    K.binary_crossentropy(true_cls, pred_cls),
                    axis=-1,
                )
            )

            xious, ious = self._bbox_xiou(pred_box, true_box)

            # xiou loss
            iou_loss1 = self._iou_norm[yolo_index] * K.sum(
                true_one * (1 - xious)
            )

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
            tf.cast(K.sum(y_true[..., self._one_index[0] :]), dtype=tf.int64)
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
