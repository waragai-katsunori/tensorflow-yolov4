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
from os import makedirs, path

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import BinaryCrossentropy, Loss, Reduction

from ..common.config import YOLOConfig


class YOLOv4Loss(Loss):
    def __init__(self, config: YOLOConfig, verbose: int = 1):
        super().__init__(name="YOLOv4Loss")

        self._bbox_xiou = {
            "iou": bbox_iou,
            "ciou": bbox_ciou,
            "giou": bbox_giou,
        }

        self._loss_config = config

        self._prob_binaryCrossentropy = BinaryCrossentropy(
            reduction=Reduction.NONE
        )

        self._verbose = verbose

    def call(self, y_true, y_pred):
        """
        @param `y_true`: Dim(batch, g_height * g_width * 3,
                                (b_x, b_y, b_w, b_h, conf, prob_0, prob_1, ...))
        @param `y_pred`: Dim(batch, g_height * g_width * 3,
                                (b_x, b_y, b_w, b_h, conf, prob_0, prob_1, ...))
        """
        yolo_name = y_pred.name.split("/")[-2]
        _, candidate_size, _ = y_pred.shape

        truth_xywh = y_true[..., 0:4]
        truth_conf = y_true[..., 4:5]
        truth_prob = y_true[..., 5:]

        pred_xywh = y_pred[..., 0:4]
        pred_conf = y_pred[..., 4:5]
        pred_prob = y_pred[..., 5:]

        one_obj = truth_conf
        num_obj = tf.reduce_sum(one_obj, axis=[1, 2])
        one_noobj = 1.0 - one_obj
        # Dim(batch, g_height * g_width * 3, 1)
        one_obj_mask = one_obj > 0.5

        # IoU Loss
        xiou = self._bbox_xiou[self._loss_config[yolo_name]["iou_loss"]](
            truth_xywh, pred_xywh
        )
        xiou_scale = 2.0 - truth_xywh[..., 2:3] * truth_xywh[..., 3:4]
        xiou_loss = one_obj * xiou_scale * (1.0 - xiou[..., tf.newaxis])
        xiou_loss = 3 * tf.reduce_mean(tf.reduce_sum(xiou_loss, axis=(1, 2)))

        # Confidence Loss
        i0 = tf.constant(0)
        zero = tf.zeros((1, candidate_size, 1), dtype=tf.float32)

        def body(i, max_iou):
            object_mask = tf.reshape(one_obj_mask[i, ...], shape=(-1,))
            truth_bbox = tf.boolean_mask(truth_xywh[i, ...], mask=object_mask)
            # g_height * g_width * 3,      1, xywh
            #                      1, answer, xywh
            #   => g_height * g_width * 3, answer
            _max_iou0 = tf.cond(
                tf.equal(num_obj[i], 0),
                lambda: zero,
                lambda: tf.reshape(
                    tf.reduce_max(
                        bbox_iou(
                            pred_xywh[i, :, tf.newaxis, :],
                            truth_bbox[tf.newaxis, ...],
                        ),
                        axis=-1,
                    ),
                    shape=(1, -1, 1),
                ),
            )
            # 1, g_height * g_width * 3, 1
            _max_iou1 = tf.cond(
                tf.equal(i, 0),
                lambda: _max_iou0,
                lambda: tf.concat([max_iou, _max_iou0], axis=0),
            )
            return tf.add(i, 1), _max_iou1

        _, max_iou = tf.while_loop(
            cond=lambda i, iou: tf.less(i, self._loss_config["net"]["batch"]),
            body=body,
            loop_vars=[i0, zero],
            shape_invariants=[
                i0.get_shape(),
                tf.TensorShape([None, candidate_size, 1]),
            ],
        )

        conf_obj_loss = one_obj * (0.0 - backend.log(pred_conf + 1e-9))
        conf_noobj_loss = (
            one_noobj
            * tf.cast(max_iou < 0.5, dtype=tf.float32)
            * (0.0 - backend.log(1.0 - pred_conf + 1e-9))
        )
        conf_loss = tf.reduce_mean(
            tf.reduce_sum(conf_obj_loss + conf_noobj_loss, axis=(1, 2))
        )

        # Probabilities Loss
        prob_loss = self._prob_binaryCrossentropy(truth_prob, pred_prob)
        prob_loss = one_obj * prob_loss[..., tf.newaxis]
        prob_loss = tf.reduce_mean(
            tf.reduce_sum(prob_loss, axis=(1, 2))
            * self._loss_config[yolo_name]["classes"]
        )

        total_loss = xiou_loss + conf_loss + prob_loss

        if self._verbose != 0:
            tf.print(
                f"{yolo_name}:",
                "iou_loss:",
                xiou_loss,
                "conf_loss:",
                conf_loss,
                "prob_loss:",
                prob_loss,
                "total_loss",
                total_loss,
            )

        return total_loss


def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1

    @return (max(a,A), max(b,B), ...)

    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + 1e-8)

    return iou


def bbox_giou(bboxes1, bboxes2):
    """
    Generalized IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1

    @return (max(a,A), max(b,B), ...)

    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + 1e-8)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-8)

    return giou


def bbox_ciou(bboxes1, bboxes2):
    """
    Complete IoU
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1

    @return (max(a,A), max(b,B), ...)

    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + 1e-8)

    enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = tf.maximum(
        bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
    )

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - rho_2 / (c_2 + 1e-8)

    v = (
        (
            tf.math.atan(bboxes1[..., 2] / (bboxes1[..., 3] + 1e-8))
            - tf.math.atan(bboxes2[..., 2] / (bboxes2[..., 3] + 1e-8))
        )
        * 2
        / 3.1415926536
    ) ** 2

    alpha = v / (1 - iou + v + 1e-8)

    ciou = diou - alpha * v

    return ciou


class YOLOCallbackAtEachStep(Callback):
    """
    Ref
        - tf.keras.callbacks.LearningRateScheduler
    """

    def __init__(self, config: YOLOConfig):
        super().__init__()
        self._cfg_burn_in = config["net"]["burn_in"]
        self._cfg_learning_rate = config["net"]["learning_rate"]
        self._cfg_max_iterations = config["net"]["max_batches"]
        self._cfg_power = config["net"]["power"]
        self._cfg_scales = config["net"]["scales"]
        self._cfg_scale_iterations = config["net"]["steps"]

    def on_train_begin(self, logs=None):
        self.model.training_iterations = 0

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_begin = batch

        self.update_lr()

    def on_train_batch_end(self, batch, logs=None):
        # compile: steps_per_execution
        # next begin iteration number
        self.model.training_iterations += batch - self._batch_begin + 1
        iterations = self.model.training_iterations
        logs = logs or {}

        logs["lr"] = backend.get_value(self.model.optimizer.lr)

        if iterations >= self._cfg_max_iterations:
            self.model.stop_training = True

    def update_lr(self):
        # on_train_batch_begin
        iterations = self.model.training_iterations

        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        lr = self._cfg_learning_rate

        # burn_in=1000
        # 0, 1, 2, ..., 999
        if iterations < self._cfg_burn_in:
            lr *= backend.pow(
                (iterations + 1) / self._cfg_burn_in, self._cfg_power
            )

        else:
            # scales=1600,1800
            # max_batches=2000
            for index, it in enumerate(
                [*self._cfg_scale_iterations, self._cfg_max_iterations]
            ):
                if iterations < it:
                    for j in range(index):
                        lr *= self._cfg_scales[j]
                    break

        if lr != float(backend.get_value(self.model.optimizer.lr)):
            tf.summary.scalar(name="learning_rate", data=lr, step=iterations)

        backend.set_value(self.model.optimizer.lr, backend.get_value(lr))


class SaveWeightsCallback(Callback):
    def __init__(
        self,
        yolo,
        dir_path: str = "trained-weights",
        step_per_save: int = 1000,
        weights_type: str = "tf",
    ):
        super().__init__()
        self._yolo = yolo
        self._weights_type = weights_type
        self._step_per_save = step_per_save

        makedirs(dir_path, exist_ok=True)

        self._path_prefix = path.join(dir_path, self._yolo.config.model_name)

        if weights_type == "tf":
            self.extension = "-checkpoint"
        else:
            self.extension = ".weights"

    def on_train_batch_end(self, batch, logs=None):
        iterations = self.model.training_iterations

        if iterations % self._step_per_save == 0:
            self._yolo.save_weights(
                "{}-{}-step{}".format(
                    self._path_prefix, iterations, self.extension
                ),
                weights_type=self._weights_type,
            )

    def on_train_end(self, logs=None):
        self._yolo.save_weights(
            "{}-final{}".format(self._path_prefix, self.extension),
            weights_type=self._weights_type,
        )
