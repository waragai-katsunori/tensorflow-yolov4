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


@tf.function
def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1

    @return (max(a,A), max(b,B), ...)

    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    xy1 = bboxes1[..., :2]
    wh_h1 = bboxes1[..., 2:4] * 0.5
    xy2 = bboxes2[..., :2]
    wh_h2 = bboxes2[..., 2:4] * 0.5

    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    lu1 = xy1 - wh_h1
    rd1 = xy1 + wh_h1
    lu2 = xy2 - wh_h2
    rd2 = xy2 + wh_h2

    left_up = K.maximum(lu1, lu2)
    right_down = K.minimum(rd1, rd2)

    inter_section = K.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + K.epsilon())

    return iou, iou


@tf.function
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
    xy1 = bboxes1[..., :2]
    wh_h1 = bboxes1[..., 2:4] * 0.5
    xy2 = bboxes2[..., :2]
    wh_h2 = bboxes2[..., 2:4] * 0.5

    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    lu1 = xy1 - wh_h1
    rd1 = xy1 + wh_h1
    lu2 = xy2 - wh_h2
    rd2 = xy2 + wh_h2

    left_up = K.maximum(lu1, lu2)
    right_down = K.minimum(rd1, rd2)

    inter_section = K.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + K.epsilon())

    enclose_left_up = K.minimum(lu1, lu2)
    enclose_right_down = K.maximum(rd1, rd2)

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - (enclose_area - union_area) / (enclose_area + K.epsilon())

    return giou, iou


@tf.function
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
    xy1 = bboxes1[..., :2]
    wh_h1 = bboxes1[..., 2:4] * 0.5
    xy2 = bboxes2[..., :2]
    wh_h2 = bboxes2[..., 2:4] * 0.5

    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    lu1 = xy1 - wh_h1
    rd1 = xy1 + wh_h1
    lu2 = xy2 - wh_h2
    rd2 = xy2 + wh_h2

    left_up = K.maximum(lu1, lu2)
    right_down = K.minimum(rd1, rd2)

    inter_section = K.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + K.epsilon())

    enclose_left_up = K.minimum(lu1, lu2)
    enclose_right_down = K.maximum(rd1, rd2)

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = K.pow(enclose_section[..., 0], 2) + K.pow(enclose_section[..., 1], 2)

    center_diagonal = xy2 - xy1

    rho_2 = K.pow(center_diagonal[..., 0], 2) + K.pow(
        center_diagonal[..., 1], 2
    )

    diou = iou - rho_2 / (c_2 + K.epsilon())

    v = K.pow(
        (
            tf.math.atan(bboxes1[..., 2] / (bboxes1[..., 3] + K.epsilon()))
            - tf.math.atan(bboxes2[..., 2] / (bboxes2[..., 3] + K.epsilon()))
        )
        * 0.636619772,  # 2/pi
        2,
    )

    alpha = v / (1 - iou + v + K.epsilon())

    ciou = diou - alpha * v

    return ciou, iou
