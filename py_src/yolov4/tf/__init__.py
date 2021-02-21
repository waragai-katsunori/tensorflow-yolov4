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
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from .dataset.keras_sequence import YOLODataset  # for exporting
from .model import YOLOv4Model
from .training.callbacks import (
    SaveWeightsCallback,  # for exporting
    YOLOCallbackAtEachStep,
)
from .training.yolo_loss import YOLOv4Loss
from .utils.mAP import create_mAP_input_files  # for exporting
from .utils.tflite import save_as_tflite  # for expoerting
from .utils.weights import (
    load_weights as _load_weights,
    save_weights as _save_weights,
)
from ..common.base_class import BaseClass

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Call tf.config.experimental.set_memory_growth(GPU0, True)")


class YOLOv4(BaseClass):
    @property
    def model(self) -> YOLOv4Model:
        return self._model

    def make_model(self):
        K.clear_session()
        _input = Input(self.config.net.input_shape)
        self._model = YOLOv4Model(config=self.config)
        self._model(_input)

    def load_weights(self, weights_path: str, weights_type: str = "tf"):
        """
        Usage:
            yolo.load_weights("checkpoints")
            yolo.load_weights("yolov4.weights", weights_type="yolo")
        """
        if weights_type == "yolo":
            _load_weights(self._model, weights_path)
        elif weights_type == "tf":
            self._model.load_weights(weights_path)

    def save_weights(
        self, weights_path: str, weights_type: str = "tf", to: int = 0
    ):
        """
        Usage:
            yolo.save_weights("checkpoints")
            yolo.save_weights("yolov4.weights", weights_type="yolo")
            yolo.save_weights("yolov4.conv.137", weights_type="yolo", to=137)
        """
        to_layer = ""
        if to > 0:
            to_layer = self.config.metalayers[to - 1].name

        if weights_type == "yolo":
            _save_weights(self._model, weights_path, to=to_layer)
        elif weights_type == "tf":
            self._model.save_weights(weights_path)

    def summary(self, line_length=90, summary_type: str = "tf", **kwargs):
        if summary_type == "tf":
            self._model.summary(line_length=line_length, **kwargs)
        else:
            self.config.summary()

    #############
    # Inference #
    #############

    @tf.function
    def _predict(self, x):
        return self._model(x, training=False)

    def predict(self, frame: np.ndarray, prob_thresh: float):
        """
        Predict one frame

        @param `frame`: Dim(height, width, channels)
        @param `prob_thresh`

        @return pred_bboxes
            Dim(-1, (x, y, w, h, cls_id, prob))
        """
        # image_data == Dim(1, input_size[1], input_size[0], channels)
        height, width, _ = frame.shape

        image_data = self.resize_image(frame)
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        candidates = self._predict(image_data)
        candidates = [
            c.numpy().astype(np.float32, copy=False) for c in candidates
        ]

        pred_bboxes = self.get_yolo_detections(
            yolos=candidates, prob_thresh=prob_thresh
        )
        self.fit_to_original(pred_bboxes, height, width)
        return pred_bboxes

    ############
    # Training #
    ############

    def compile(
        self,
        optimizer=None,
        loss=None,
        **kwargs,
    ):
        if optimizer is None:
            optimizer = Adam(learning_rate=self.config.net.learning_rate)

        if loss is None:
            loss = YOLOv4Loss(config=self.config, model=self.model)

        return self._model.compile(
            optimizer=optimizer,
            loss=loss,
            **kwargs,
        )

    def fit(
        self,
        dataset,
        callbacks=None,
        validation_data=None,
        validation_steps=None,
        verbose: int = 3,
        **kwargs,
    ):
        """
        verbose=3 is one line per step
        """
        callbacks = callbacks or []
        callbacks.append(
            YOLOCallbackAtEachStep(config=self.config, verbose=verbose)
        )

        epochs = self.config.net.max_batches // len(dataset) + 1

        return self._model.fit(
            dataset,
            epochs=epochs,
            verbose=verbose if verbose < 3 else 0,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            **kwargs,
        )
