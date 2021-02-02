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
from typing import Any, Dict, List

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    from tflite_runtime.interpreter import load_delegate
except ModuleNotFoundError:
    import tensorflow.lite as tflite

    load_delegate = tflite.experimental.load_delegate

from ..common.base_class import BaseClass
from ..common.config import YOLOConfig

EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]


class YOLOv4(BaseClass):
    def __init__(self, tpu: bool = False):
        super().__init__()
        self._input_details: Dict[str, Any]
        self._interpreter: tflite.Interpreter
        self._output_details: List[Dict[str, Any]]
        self._tpu = tpu

    @property
    def tpu(self) -> bool:
        return self._tpu

    def load_tflite(
        self, tflite_path: str, edgetpu_lib: str = EDGETPU_SHARED_LIB
    ) -> None:
        if self.tpu:
            self._interpreter = tflite.Interpreter(
                model_path=tflite_path,
                experimental_delegates=[load_delegate(edgetpu_lib)],
            )
        else:
            self._interpreter = tflite.Interpreter(model_path=tflite_path)
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()[0]
        if (
            input_details["shape"][1] != self.config.input_shape[0]
            or input_details["shape"][2] != self.config.input_shape[1]
            or input_details["shape"][3] != self.config.input_shape[2]
        ):
            raise RuntimeError(
                "YOLOv4: config.input_shape and tflite.input_details['shape']"
                " do not match."
            )
        self._input_details = input_details

        self._output_details = self._interpreter.get_output_details()
        self.config.output_shape = tuple(
            output_detail["shape"] for output_detail in self._output_details
        )

        if not self.config.with_head:
            self._head = tuple(
                YOLOv3Head(
                    config=self.config, name=f"yolo{i}", grid_shape=grid_shape
                )
                for i, grid_shape in enumerate(self.config.output_shape)
            )

    #############
    # Inference #
    #############

    def _predict(self, x):
        self._interpreter.set_tensor(self._input_details["index"], x)
        self._interpreter.invoke()

        candidates = [
            self._interpreter.get_tensor(output_detail["index"])
            if output_detail["dtype"] is np.float32
            else (
                self._interpreter.get_tensor(output_detail["index"]).astype(
                    np.float32
                )
                / 255.0
            )
            for output_detail in self._output_details
        ]

        if self.config.with_head:
            return np.concatenate(candidates, axis=1)

        return np.concatenate(
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
        image_data = self.resize_image(frame)
        if self._input_details["dtype"] is np.float32:
            image_data = image_data.astype(np.float32) / 255.0
        image_data = image_data[np.newaxis, ...]

        candidates = self._predict(image_data)

        pred_bboxes = self.candidates_to_pred_bboxes(
            candidates[0],
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        pred_bboxes = self.fit_pred_bboxes_to_original(pred_bboxes, frame.shape)
        return pred_bboxes


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class YOLOv3Head:
    def __init__(self, config: YOLOConfig, name: str, grid_shape):
        self._anchors = tuple(
            config[name]["anchors"][mask] for mask in config[name]["mask"]
        )

        _, grid_height, grid_width, filters = grid_shape

        grid_coord = []
        for y in range(grid_height):
            grid_coord.append([])
            for x in range(grid_width):
                grid_coord[y].append((x / grid_width, y / grid_height))
            grid_coord[y] = tuple(grid_coord[y])

        self._grid_coord = tuple(grid_coord)

        self._inver_grid_wh = (1 / grid_width, 1 / grid_height)

        self._inver_image_wh = (
            1 / config["net"]["width"],
            1 / config["net"]["height"],
        )

        self._return_shape = (-1, grid_height * grid_width, filters // 3)

        self._scale_x_y = config[name]["scale_x_y"]

    def __call__(self, x):
        raw_split = np.split(x, 3, axis=-1)

        sig = sigmoid(x)
        sig_split = np.split(sig, 3, axis=-1)

        output = []
        for i in range(3):

            # Operation not supported on Edge TPU
            xy, _, oc = np.split(sig_split[i], [2, 4], axis=-1)
            _, wh, _ = np.split(raw_split[i], [2, 4], axis=-1)
            wh = np.exp(wh)

            # Can be Mapped to Edge TPU
            if self._scale_x_y != 1.0:
                xy = (xy - 0.5) * self._scale_x_y + 0.5
            xy *= self._inver_grid_wh
            xy += self._grid_coord

            wh = wh * self._anchors[i] * self._inver_image_wh

            output.append(
                np.reshape(
                    np.concatenate([xy, wh, oc], axis=-1), self._return_shape
                )
            )

        return np.concatenate(output, axis=1)
