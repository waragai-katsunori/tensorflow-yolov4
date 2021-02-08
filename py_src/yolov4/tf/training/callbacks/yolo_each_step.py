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
import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

from ....common.config import YOLOConfig


class YOLOCallbackAtEachStep(Callback):
    """
    Ref
        - tf.keras.callbacks.LearningRateScheduler
    """

    def __init__(self, config: YOLOConfig, verbose: int):
        super().__init__()
        self._cfg_burn_in = config.net.burn_in
        self._cfg_learning_rate = config.net.learning_rate
        self._cfg_max_step = config.net.max_batches
        self._cfg_power = config.net.power
        self._cfg_scales = config.net.scales
        self._cfg_scale_steps = config.net.steps

        self._verbose = verbose

    def on_train_begin(self, logs=None):
        self.model._iou_loss = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.model._obj_loss = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.model._cls_loss = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.model._total_loss = tf.Variable(
            0, dtype=tf.float32, trainable=False
        )

        self.model._total_truth = tf.Variable(
            0, dtype=tf.int64, trainable=False
        )
        self.model._ious = tf.Variable(0, dtype=tf.float32, trainable=False)
        self.model._recall50 = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.model._recall75 = tf.Variable(0, dtype=tf.int32, trainable=False)

    def on_train_batch_begin(self, batch, logs=None):
        self.model._iou_loss.assign(0)
        self.model._obj_loss.assign(0)
        self.model._cls_loss.assign(0)
        self.model._total_loss.assign(0)

        self._prev_total_truth = self.model._total_truth.value()
        self.model._ious.assign(0)
        self.model._recall50.assign(0)
        self.model._recall75.assign(0)

        self.start_time = time.time()
        self.update_lr()

    def on_train_batch_end(self, batch, logs=None):
        # compile: steps_per_execution
        # next begin iteration number
        step = self.model._train_counter
        spe = self.model._steps_per_execution
        spe_f = tf.cast(spe, dtype=tf.float32)

        leraning_rate = K.get_value(self.model.optimizer.lr)

        iou_loss = self.model._iou_loss.value() / spe_f
        obj_loss = self.model._obj_loss.value() / spe_f
        cls_loss = self.model._cls_loss.value() / spe_f
        total_loss = self.model._total_loss.value() / spe_f

        total_truth = self.model._total_truth.value()
        truth = total_truth - self._prev_total_truth
        truth_f = tf.cast(truth, dtype=tf.float32)
        iou = self.model._ious.value() / truth_f
        recall50 = self.model._recall50.value()
        recall50_f = tf.cast(recall50, dtype=tf.float32) / truth_f
        recall75 = self.model._recall75.value()
        recall75_f = tf.cast(recall75, dtype=tf.float32) / truth_f

        tf.summary.scalar(name="iou_loss", data=iou_loss, step=step)
        tf.summary.scalar(name="obj_loss", data=obj_loss, step=step)
        tf.summary.scalar(name="cls_loss", data=cls_loss, step=step)
        tf.summary.scalar(name="total_loss", data=total_loss, step=step)

        tf.summary.scalar(name="total_truth", data=total_truth, step=step)
        tf.summary.scalar(name="truth", data=truth, step=step)
        tf.summary.scalar(name="iou", data=iou, step=step)
        tf.summary.scalar(name="recall50", data=recall50_f, step=step)
        tf.summary.scalar(name="recall75", data=recall75_f, step=step)

        if self._verbose == 3:
            verbose = f"step: {step.numpy()}, "
            verbose += f"{time.time()-self.start_time:6.2f}s, "
            verbose += f"Truth: {truth.numpy():4}, "
            verbose += f"Avg IOU: {iou.numpy():5.3f}, "
            verbose += f".5R: {recall50_f.numpy():5.3f}, "
            verbose += f".75R: {recall75_f.numpy():5.3f}, "
            verbose += f"lr: {leraning_rate:7.5f}, "
            verbose += "Loss => "
            verbose += f"IOU: {iou_loss.numpy():6.3f}, "
            verbose += f"OBJ: {obj_loss.numpy():6.3f}, "
            verbose += f"CLS: {cls_loss.numpy():6.3f}, "
            verbose += f"Total: {total_loss.numpy():6.3f}"
            print(verbose)

        logs = logs or {}
        logs["lr"] = leraning_rate
        if step >= self._cfg_max_step:
            self.model.stop_training = True

    def update_lr(self):
        # on_train_batch_begin
        step = self.model._train_counter

        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        lr = self._cfg_learning_rate

        # burn_in=1000
        # 0, 1, 2, ..., 999
        if step < self._cfg_burn_in:
            lr *= K.pow((step + 1) / self._cfg_burn_in, self._cfg_power)

        else:
            # scales=1600,1800
            # max_batches=2000
            for index, it in enumerate(
                [*self._cfg_scale_steps, self._cfg_max_step]
            ):
                if step < it:
                    for j in range(index):
                        lr *= self._cfg_scales[j]
                    break

        if lr != float(K.get_value(self.model.optimizer.lr)):
            tf.summary.scalar(name="learning_rate", data=lr, step=step)

        K.set_value(self.model.optimizer.lr, K.get_value(lr))
