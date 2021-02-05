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
import shutil

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import weights
from .dataset import YOLODataset
from .model import YOLOv3Head, YOLOv4Model
from .train import YOLOCallbackAtEachStep, YOLOv4Loss
from ..common.base_class import BaseClass

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Call tf.config.experimental.set_memory_growth(GPU0, True)")


class YOLOv4(BaseClass):
    def __init__(self):
        super().__init__()
        self._model = None

    def make_model(self):
        keras.backend.clear_session()
        _input = keras.layers.Input(self.config.net.input_shape)
        self._model = YOLOv4Model(self.config)
        self._model(_input)

        if not self.config.with_head:
            self._head = tuple(
                YOLOv3Head(config=self.config, name=f"yolo{i}")
                for i in range(self.config.count["yolo"])
            )

    def load_weights(self, weights_path: str, weights_type: str = "tf"):
        """
        Usage:
            yolo.load_weights("checkpoints")
            yolo.load_weights("yolov4.weights", weights_type="yolo")
        """
        if weights_type == "yolo":
            weights.load_weights(self._model, weights_path)
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
            for name, option in self.config.items():
                if option["count"] == to - 1:
                    to_layer = name
                    break

        if weights_type == "yolo":
            weights.save_weights(self._model, weights_path, to=to_layer)
        elif weights_type == "tf":
            self._model.save_weights(weights_path)

    def save_as_tflite(
        self,
        tflite_path: str,
        quantization: str = "",
        dataset: YOLODataset = None,
        num_calibration_steps: int = 100,
    ):
        """
        Save model and weights as tflite

        Usage:
            yolo.save_as_tflite("yolov4.tflite")
            yolo.save_as_tflite("yolov4-float16.tflite", "float16")
            yolo.save_as_tflite("yolov4-int.tflite", "int", dataset)
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self._model)

        _supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

        def representative_dataset_gen():
            count = 0
            while True:
                for images, _ in dataset:
                    for i in range(len(images)):
                        yield [tf.cast(images[i : i + 1, ...], tf.float32)]
                        count += 1
                        if count >= num_calibration_steps:
                            return

        if quantization != "":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
        elif quantization == "int":
            converter.representative_dataset = representative_dataset_gen
        elif quantization == "full_int8":
            converter.experimental_new_converter = False
            converter.representative_dataset = representative_dataset_gen
            _supported_ops += [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32
        else:
            raise ValueError(
                f"YOLOv4: '{quantization}' is not a valid quantization"
            )

        converter.target_spec.supported_ops = _supported_ops

        tflite_model = converter.convert()
        with tf.io.gfile.GFile(tflite_path, "wb") as fd:
            fd.write(tflite_model)

    def summary(self, summary_type: str = "tf"):
        if summary_type == "tf":
            self._model.summary()
        else:
            self.config.summary()

    #############
    # Inference #
    #############

    @tf.function
    def _predict(self, x):
        candidates = self._model(x, training=False)
        # [yolo0, yolo1, ...]
        # yolo == Dim(1, output_size * output_size * anchors, (bbox))
        if self.config.with_head:
            return tf.concat(candidates, axis=1)

        return tf.concat(
            [head(candidates[i]) for i, head in enumerate(self._head)],
            axis=1,
        )

    def predict(
        self,
        frame: np.ndarray,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.25,
    ):
        """
        Predict one frame

        @param frame: Dim(height, width, channels)

        @return pred_bboxes == Dim(-1, (x, y, w, h, class_id, probability))
        """
        # image_data == Dim(1, input_size[1], input_size[0], channels)
        image_data = self.resize_image(frame)
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        candidates = self._predict(image_data)

        # Select 0
        pred_bboxes = self.candidates_to_pred_bboxes(
            candidates[0].numpy(),
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        pred_bboxes = self.fit_pred_bboxes_to_original(pred_bboxes, frame.shape)
        return pred_bboxes

    ############
    # Training #
    ############

    def load_dataset(
        self,
        dataset_path: str,
        dataset_type: str = "converted_coco",
        image_path_prefix: str = "",
        training: bool = False,
    ) -> YOLODataset:
        return YOLODataset(
            config=self.config,
            dataset_path=dataset_path,
            dataset_type=dataset_type,
            image_path_prefix=image_path_prefix,
            training=training,
        )

    def compile(
        self,
        loss_verbose: int = 1,
        optimizer=None,
        **kwargs,
    ):
        # TODO: steps_per_execution tensorflow2.4.0-rc4

        if optimizer is None:
            optimizer = keras.optimizers.Adam(
                learning_rate=self.config.net.learning_rate
            )

        self._model.compile(
            optimizer=optimizer,
            loss=YOLOv4Loss(config=self.config, verbose=loss_verbose),
            **kwargs,
        )

    def fit(
        self,
        dataset,
        callbacks=None,
        validation_data=None,
        validation_steps=None,
        verbose: int = 2,
        **kwargs,
    ):
        callbacks = callbacks or []
        callbacks.append(YOLOCallbackAtEachStep(config=self.config))

        epochs = self.config.net.max_batches // len(dataset) + 1

        return self._model.fit(
            dataset,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            **kwargs,
        )

    def save_dataset_for_mAP(
        self,
        dataset: YOLODataset,
        mAP_path: str,
        images_optional: bool = False,
        num_sample: int = None,
    ):
        """
        gt: name left top right bottom
        dr: name confidence left top right bottom

        @parma `dataset`
        @param `mAP_path`
        @parma `images_optional`: If `True`, images are copied to the
                `mAP_path`.
        @param `num_sample`: Number of images for mAP. If `None`, all images in
                `data_set` are used.
        """
        input_path = path.join(mAP_path, "input")

        if path.exists(input_path):
            shutil.rmtree(input_path)
        makedirs(input_path)

        gt_dir_path = path.join(input_path, "ground-truth")
        dr_dir_path = path.join(input_path, "detection-results")
        makedirs(gt_dir_path)
        makedirs(dr_dir_path)

        img_dir_path = ""
        if images_optional:
            img_dir_path = path.join(input_path, "images-optional")
            makedirs(img_dir_path)

        max_dataset_size = len(dataset)

        if num_sample is None:
            num_sample = max_dataset_size

        for i in range(num_sample):
            # image_path, [[x, y, w, h, class_id], ...]
            _dataset = dataset._dataset[i % max_dataset_size].copy()

            if images_optional:
                image_path = path.join(img_dir_path, "image_{}.jpg".format(i))
                shutil.copy(_dataset[0], image_path)

            image = cv2.imread(_dataset[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape

            _dataset[1] = _dataset[1] * np.array(
                [width, height, width, height, 1]
            )

            # ground-truth
            with open(
                path.join(gt_dir_path, "image_{}.txt".format(i)),
                "w",
            ) as fd:
                for xywhc in _dataset[1]:
                    # name left top right bottom
                    class_name = self.config.names[int(xywhc[4])].replace(
                        " ", "_"
                    )
                    left = int(xywhc[0] - xywhc[2] / 2)
                    top = int(xywhc[1] - xywhc[3] / 2)
                    right = int(xywhc[0] + xywhc[2] / 2)
                    bottom = int(xywhc[1] + xywhc[3] / 2)
                    fd.write(
                        "{} {} {} {} {}\n".format(
                            class_name, left, top, right, bottom
                        )
                    )

            pred_bboxes = self.predict(image)
            pred_bboxes = pred_bboxes * np.array(
                [width, height, width, height, 1, 1]
            )

            # detection-results
            with open(
                path.join(dr_dir_path, "image_{}.txt".format(i)),
                "w",
            ) as fd:
                for xywhcp in pred_bboxes:
                    # name confidence left top right bottom
                    class_name = self.config.names[int(xywhcp[4])].replace(
                        " ", "_"
                    )
                    probability = xywhcp[5]
                    left = int(xywhcp[0] - xywhcp[2] / 2)
                    top = int(xywhcp[1] - xywhcp[3] / 2)
                    right = int(xywhcp[0] + xywhcp[2] / 2)
                    bottom = int(xywhcp[1] + xywhcp[3] / 2)
                    fd.write(
                        "{} {} {} {} {} {}\n".format(
                            class_name, probability, left, top, right, bottom
                        )
                    )
