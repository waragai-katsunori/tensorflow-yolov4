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
import colorsys
from typing import Dict, Union

# Don't import tensorflow
import cv2
import numpy as np

_MAX_CLASSES = 14 * 6
_HSV = [(x / _MAX_CLASSES, 1.0, 1.0) for x in range(int(_MAX_CLASSES * 1.2))]
_COLORS = [colorsys.hsv_to_rgb(*x) for x in _HSV]
_COLORS = [(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in _COLORS]
_BBOX_COLORS = []
for i in range(_MAX_CLASSES):
    # 0 14 28 42 56 70 1 15 29 43 57 71 2 ...
    _BBOX_COLORS.append(_COLORS[14 * (i % 6) + (i // 6)])


def resize_image(
    image: np.ndarray,
    target_shape: Union[list, tuple],
    ground_truth: np.ndarray = None,
):
    """
    @param `image`:        Dim(height, width, channels)
    @param `target_shape`:  (height, width, ...)
    @param `ground_truth`: [[center_x, center_y, w, h, class_id], ...]

    @return resized_image or (resized_image, resized_ground_truth)

    Usage:
        image = media.resize_image(image, yolo.input_size)
        image, ground_truth = media.resize_image(image, yolo.input_size,
                                                                ground_truth)
    """
    height, width, _ = image.shape
    target_height = target_shape[0]
    target_width = target_shape[1]

    if width / height >= target_width / target_height:
        scale = target_width / width
    else:
        scale = target_height / height

    # Resize
    if scale != 1:
        width = int(round(width * scale))
        height = int(round(height * scale))
        resized_image = cv2.resize(image, (width, height))
    else:
        resized_image = np.copy(image)

    # Pad
    dw = target_width - width
    dh = target_height - height

    if not (dw == 0 and dh == 0):
        dw = dw // 2
        dh = dh // 2
        # height, width, channels
        padded_image = np.full(
            (target_height, target_width, 3), 255, dtype=np.uint8
        )
        padded_image[dh : height + dh, dw : width + dw, :] = resized_image
    else:
        padded_image = resized_image

    if ground_truth is None:
        return padded_image

    # Resize ground truth
    ground_truth = np.copy(ground_truth)

    if dw > dh:
        scale = width / target_width
        ground_truth[:, 0] = scale * (ground_truth[:, 0] - 0.5) + 0.5
        ground_truth[:, 2] = scale * ground_truth[:, 2]
    elif dw < dh:
        scale = height / target_height
        ground_truth[:, 1] = scale * (ground_truth[:, 1] - 0.5) + 0.5
        ground_truth[:, 3] = scale * ground_truth[:, 3]

    return padded_image, ground_truth


def draw_bboxes(
    image: np.ndarray, pred_bboxes: np.ndarray, names: Dict[int, str]
):
    """
    @parma `image`:  Dim(height, width, channel)
    @parma `pred_bboxes`
        Dim(-1, (x, y, w, h, cls_id, prob))

    @return drawn_image

    Usage:
        image = yolo.draw_bboxes(image, bboxes)
    """
    image = np.copy(image)
    height, width, _ = image.shape

    bboxes = pred_bboxes * np.array([width, height, width, height, 1, 1])

    # Draw bboxes
    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        half_w = int(bbox[2] / 2)
        half_h = int(bbox[3] / 2)

        font_size = 0.4
        font_thickness = 1

        cls_id = int(bbox[4])
        prob = bbox[5]
        color = _BBOX_COLORS[cls_id]

        # Get text size
        bbox_text = "{}: {:.1%}".format(names[cls_id], prob)
        t_w, t_h = cv2.getTextSize(bbox_text, 0, font_size, font_thickness)[0]
        t_h += 3

        # Draw box
        top = c_y - half_h
        if top < t_h:
            top = t_h
        left = c_x - half_w
        if left < 1:
            left = 1
        bottom = c_y + half_h
        if bottom >= height:
            bottom = height - 1
        right = c_x + half_w
        if right >= width:
            right = width - 1

        cv2.rectangle(image, (left, top), (right, bottom), color, 1)

        # Draw text box
        cv2.rectangle(image, (left, top), (left + t_w, top - t_h), color, -1)

        # Draw text
        cv2.putText(
            image,
            bbox_text,
            (left, top - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (
                255 - color[0],
                255 - color[1],
                255 - color[2],
            ),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return image
