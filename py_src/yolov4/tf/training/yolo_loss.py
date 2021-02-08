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


class YOLOv4Loss(Loss):
    def __init__(self, config: YOLOConfig, model: YOLOv4Model):
        super().__init__(name="YOLOv4Loss")
        self._metayolos = []
        for i in range(config.layer_count["yolo"]):
            self._metayolos.append(config.find_metalayer("yolo", i))
        self._metanet = config.net

        self.model = model

        self._bbox_xiou = {
            "iou": bbox_iou,
            "ciou": bbox_ciou,
            "giou": bbox_giou,
        }

    def call(self, y_true, y_pred):
        """
        @param `y_true`: Dim(batch, yolo.h, yolo.w, yolo.c)
        @param `y_pred`: Dim(batch, yolo.h, yolo.w, yolo.c + len(mask))
        """
        yolo_name = y_pred.name.split("/")[-2]
        yolo_index = int(yolo_name.split("_")[-1])
        metayolo = self._metayolos[yolo_index]

        bbox_xiou = self._bbox_xiou[metayolo.iou_loss]
        classes = tf.constant(metayolo.classes)
        cls_normalizer = tf.constant(metayolo.cls_normalizer)
        iou_normalizer = tf.constant(metayolo.iou_normalizer)
        mask = tf.convert_to_tensor(metayolo.mask)
        obj_normalizer = tf.constant(metayolo.obj_normalizer)

        def batch_loop(batch, iou_loss0, obj_loss0, cls_loss0):
            # y_pred0 == Dim(height, width, filters)
            # y_true0 == Dim(height, width, filters + len(mask))
            y_true0 = y_true[batch, ...]
            y_pred0 = y_pred[batch, ...]

            def anchor_loop(anchor, iou_loss1, obj_loss1, cls_loss1):
                stride = 5 + classes
                box_index = anchor * stride
                obj_index = box_index + 4
                cls_index = box_index + 5
                next_box_index = box_index + stride
                one_index = mask.shape[0] * stride + anchor

                true_box = y_true0[..., box_index:obj_index]
                true_obj = y_true0[..., obj_index]
                true_cls = y_true0[..., cls_index:next_box_index]
                true_one = y_true0[..., one_index]

                pred_box = y_pred0[..., box_index:obj_index]
                pred_obj = y_pred0[..., obj_index]
                pred_cls = y_pred0[..., cls_index:next_box_index]

                xious, ious = bbox_xiou(pred_box, true_box)

                # xiou loss
                iou_loss2 = iou_normalizer * K.sum(true_one * (1 - xious))

                # metrics update
                ious = true_one * ious
                self.model._total_truth.assign_add(
                    tf.cast(K.sum(true_one), dtype=tf.int64)
                )
                self.model._ious.assign_add(K.sum(ious))
                self.model._recall50.assign_add(
                    K.sum(tf.cast(ious > 0.5, dtype=tf.int32))
                )
                self.model._recall75.assign_add(
                    K.sum(tf.cast(ious > 0.75, dtype=tf.int32))
                )

                # obj loss
                obj_loss2 = obj_normalizer * K.sum(
                    K.binary_crossentropy(true_obj, pred_obj)
                )

                # cls loss
                cls_loss2 = cls_normalizer * K.sum(
                    true_one
                    * K.sum(
                        K.binary_crossentropy(true_cls, pred_cls),
                        axis=-1,
                    )
                )

                return (
                    tf.add(anchor, 1),
                    tf.add(iou_loss1, iou_loss2),
                    tf.add(obj_loss1, obj_loss2),
                    tf.add(cls_loss1, cls_loss2),
                )

            anchor0 = tf.constant(0)
            iou_loss1 = tf.constant(0, dtype=tf.float32)
            obj_loss1 = tf.constant(0, dtype=tf.float32)
            cls_loss1 = tf.constant(0, dtype=tf.float32)
            _, iou_loss1, obj_loss1, cls_loss1 = tf.while_loop(
                cond=lambda batch, _, __, ___: tf.less(batch, mask.shape[0]),
                body=anchor_loop,
                loop_vars=[anchor0, iou_loss1, obj_loss1, cls_loss1],
            )

            return (
                tf.add(batch, 1),
                tf.add(iou_loss0, iou_loss1),
                tf.add(obj_loss0, obj_loss1),
                tf.add(cls_loss0, cls_loss1),
            )

        batch0 = tf.constant(0)
        iou_loss0 = tf.constant(0, dtype=tf.float32)
        obj_loss0 = tf.constant(0, dtype=tf.float32)
        cls_loss0 = tf.constant(0, dtype=tf.float32)
        _, iou_loss0, obj_loss0, cls_loss0 = tf.while_loop(
            cond=lambda batch, _, __, ___: tf.less(batch, self._metanet.batch),
            body=batch_loop,
            loop_vars=[batch0, iou_loss0, obj_loss0, cls_loss0],
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
