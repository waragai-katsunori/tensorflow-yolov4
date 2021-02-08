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

    bboxes1_coor = K.concatenate(
        [
            bboxes1[..., :2] - bboxes1[..., 2:4] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:4] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = K.concatenate(
        [
            bboxes2[..., :2] - bboxes2[..., 2:4] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:4] * 0.5,
        ],
        axis=-1,
    )

    left_up = K.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = K.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = K.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + K.epsilon())

    return iou, iou


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

    bboxes1_coor = K.concatenate(
        [
            bboxes1[..., :2] - bboxes1[..., 2:4] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:4] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = K.concatenate(
        [
            bboxes2[..., :2] - bboxes2[..., 2:4] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:4] * 0.5,
        ],
        axis=-1,
    )

    left_up = K.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = K.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = K.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + K.epsilon())

    enclose_left_up = K.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = K.maximum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    enclose_section = enclose_right_down - enclose_left_up
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    giou = iou - (enclose_area - union_area) / (enclose_area + K.epsilon())

    return giou, iou


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

    bboxes1_coor = K.concatenate(
        [
            bboxes1[..., :2] - bboxes1[..., 2:4] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:4] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = K.concatenate(
        [
            bboxes2[..., :2] - bboxes2[..., 2:4] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:4] * 0.5,
        ],
        axis=-1,
    )

    left_up = K.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = K.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = K.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = inter_area / (union_area + K.epsilon())

    enclose_left_up = K.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    enclose_right_down = K.maximum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    enclose_section = enclose_right_down - enclose_left_up

    c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

    center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

    rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

    diou = iou - rho_2 / (c_2 + K.epsilon())

    v = (
        (
            tf.math.atan(bboxes1[..., 2] / (bboxes1[..., 3] + K.epsilon()))
            - tf.math.atan(bboxes2[..., 2] / (bboxes2[..., 3] + K.epsilon()))
        )
        * 2
        / 3.1415926536
    ) ** 2

    alpha = v / (1 - iou + v + K.epsilon())

    ciou = diou - alpha * v

    return ciou, iou
