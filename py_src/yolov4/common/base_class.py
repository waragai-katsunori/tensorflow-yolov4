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
from os import path
import time

import cv2
import numpy as np

from . import media, predict
from .config import YOLOConfig


class BaseClass:
    def __init__(self):
        self.config = YOLOConfig()

    def resize_image(self, image, ground_truth=None):
        """
        @param image:        Dim(height, width, channels)
        @param ground_truth: [[center_x, center_y, w, h, class_id], ...]

        @return resized_image or (resized_image, resized_ground_truth)

        Usage:
            image = yolo.resize_image(image)
            image, ground_truth = yolo.resize_image(image, ground_truth)
        """
        return media.resize_image(
            image,
            target_shape=self.config.input_shape,
            ground_truth=ground_truth,
        )

    def candidates_to_pred_bboxes(
        self,
        candidates,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.25,
    ):
        """
        @param candidates: Dim(-1, (x, y, w, h, conf, prob_0, prob_1, ...))

        @return Dim(-1, (x, y, w, h, class_id, probability))
        """
        return predict.candidates_to_pred_bboxes(
            candidates,
            input_shape=self.config.input_shape,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

    def fit_pred_bboxes_to_original(self, pred_bboxes, original_shape):
        """
        @param pred_bboxes:    Dim(-1, (x, y, w, h, class_id, probability))
        @param original_shape: (height, width, channels)
        """
        return predict.fit_pred_bboxes_to_original(
            pred_bboxes,
            input_shape=self.config.input_shape,
            original_shape=original_shape,
        )

    def draw_bboxes(self, image, bboxes):
        """
        @parma image:  Dim(height, width, channel)
        @param bboxes: (candidates, 4) or (candidates, 5)
                [[center_x, center_y, w, h, class_id], ...]
                [[center_x, center_y, w, h, class_id, propability], ...]

        @return drawn_image

        Usage:
            image = yolo.draw_bboxes(image, bboxes)
        """
        return media.draw_bboxes(image, bboxes, names=self.config.names)

    #############
    # Inference #
    #############

    def predict(
        self,
        frame: np.ndarray,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.25,
    ):
        # pylint: disable=unused-argument, no-self-use
        return [[0.0, 0.0, 0.0, 0.0, -1]]

    def inference(
        self,
        media_path,
        is_image: bool = True,
        cv_apiPreference=None,
        cv_frame_size: tuple = None,
        cv_fourcc: str = None,
        cv_waitKey_delay: int = 1,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.25,
    ):
        if isinstance(media_path, str) and not path.exists(media_path):
            raise FileNotFoundError("{} does not exist".format(media_path))

        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

        if is_image:
            frame = cv2.imread(media_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            bboxes = self.predict(
                frame,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
            exec_time = time.time() - start_time
            print("time: {:.2f} ms".format(exec_time * 1000))

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image = self.draw_bboxes(frame, bboxes)
            cv2.imshow("result", image)
        else:
            if cv_apiPreference is None:
                cap = cv2.VideoCapture(media_path)
            else:
                cap = cv2.VideoCapture(media_path, cv_apiPreference)

            if cv_frame_size is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cv_frame_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv_frame_size[1])

            if cv_fourcc is not None:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cv_fourcc))

            prev_time = time.time()
            if cap.isOpened():
                while True:
                    try:
                        is_success, frame = cap.read()
                    except cv2.error:
                        continue

                    if not is_success:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    predict_start_time = time.time()
                    bboxes = self.predict(
                        frame,
                        iou_threshold=iou_threshold,
                        score_threshold=score_threshold,
                    )
                    predict_exec_time = time.time() - predict_start_time

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    image = self.draw_bboxes(frame, bboxes)
                    curr_time = time.time()

                    cv2.putText(
                        image,
                        "preidct: {:.2f} ms, fps: {:.2f}".format(
                            predict_exec_time * 1000,
                            1 / (curr_time - prev_time),
                        ),
                        org=(5, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6,
                        color=(50, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    prev_time = curr_time

                    cv2.imshow("result", image)
                    if cv2.waitKey(cv_waitKey_delay) & 0xFF == ord("q"):
                        break

            cap.release()

        print("YOLOv4: Inference is finished")
        while cv2.waitKey(10) & 0xFF != ord("q"):
            pass
        cv2.destroyWindow("result")
