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
import platform
from typing import List

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    from tflite_runtime.interpreter import load_delegate
except ModuleNotFoundError:
    import tensorflow.lite as tflite

    load_delegate = tflite.experimental.load_delegate

from ..common.base_class import BaseClass
from ..common import (
    yolo_tpu_layer as _yolo_tpu_layer,
    yolo_tpu_layer_new_coords as _yolo_tpu_layer_new_coords,
)

EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


class YOLOv4(BaseClass):
    def load_tflite(
        self, tflite_path: str, edgetpu_lib: str = EDGETPU_SHARED_LIB
    ) -> None:
        self._tpu = self.config.layer_count["yolo_tpu"] > 0

        if self._tpu:
            self._interpreter = tflite.Interpreter(
                model_path=tflite_path,
                experimental_delegates=[load_delegate(edgetpu_lib)],
            )
        else:
            self._interpreter = tflite.Interpreter(model_path=tflite_path)

        self._interpreter.allocate_tensors()

        # input_details
        input_details = self._interpreter.get_input_details()[0]
        if (
            input_details["shape"][1] != self.config.net.input_shape[0]
            or input_details["shape"][2] != self.config.net.input_shape[1]
            or input_details["shape"][3] != self.config.net.input_shape[2]
        ):
            raise RuntimeError(
                "YOLOv4: config.input_shape and tflite.input_details['shape']"
                " do not match."
            )
        self._input_details = input_details
        self._input_float = self._input_details["dtype"] is np.float32

        # output_details
        self._output_details = self._interpreter.get_output_details()

        self._num_masks = len(self.config.metayolos[-1].mask)
        self._new_coords = self.config.metayolos[-1].new_coords

        self._scale_x_y = []
        for metayolo in self.config.metayolos:
            self._scale_x_y.append(metayolo.scale_x_y)

    def summary(self):
        self.config.summary()

    #############
    # Inference #
    #############

    def _predict(self, x: np.ndarray) -> List[np.ndarray]:
        self._interpreter.set_tensor(self._input_details["index"], x)
        self._interpreter.invoke()
        # [yolo0, yolo1, ...]
        # yolo == Dim(1, height, width, channels)
        # yolo_tpu == x, logistic(x)

        yolos = [
            self._interpreter.get_tensor(output_detail["index"])
            for output_detail in self._output_details
        ]

        if self._tpu:
            _yolos = []
            if self._new_coords:
                for i, scale_x_y in enumerate(self._scale_x_y):
                    _yolo_tpu_layer_new_coords(
                        yolos[i], self._num_masks, scale_x_y
                    )
                    _yolos.append(yolos[i])
            else:
                for i, scale_x_y in enumerate(self._scale_x_y):
                    _yolo_tpu_layer(
                        yolos[2 * i],
                        yolos[2 * i + 1],
                        self._num_masks,
                        scale_x_y,
                    )
                    _yolos.append(yolos[2 * i + 1])

            return _yolos

        return yolos

    def predict(self, frame: np.ndarray, prob_thresh: float) -> np.ndarray:
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
        if self._input_float:
            candidates = self._predict(
                image_data[np.newaxis, ...].astype(np.float32) / 255
            )
        else:
            candidates = self._predict(image_data[np.newaxis, ...])

        pred_bboxes = self.get_yolo_detections(
            yolos=candidates, prob_thresh=prob_thresh
        )
        self.fit_to_original(pred_bboxes, height, width)
        return pred_bboxes
