/*
 * MIT License
 *
 * Copyright (c) 2021 Hyeonki Hong <hhk7734@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::list convert_dataset_to_ground_truth(py::array_t<float> &dataset,
                                         py::array_t<float> &metayolos,
                                         py::array_t<float> &anchors);

py::array_t<float> get_yolo_detections(py::array_t<float> &yolo_0,
                                       py::array_t<float> &yolo_1,
                                       py::array_t<float> &yolo_2,
                                       py::array_t<int> &  mask_0,
                                       py::array_t<int> &  mask_1,
                                       py::array_t<int> &  mask_2,
                                       py::array_t<float> &anchors,
                                       float               beta_nms,
                                       bool                new_coords);

py::array_t<float> get_yolo_tiny_detections(py::array_t<float> &yolo_0,
                                            py::array_t<float> &yolo_1,
                                            py::array_t<int> &  mask_0,
                                            py::array_t<int> &  mask_1,
                                            py::array_t<float> &anchors,
                                            float               beta_nms,
                                            bool                new_coords);

void fit_to_original(py::array_t<float> &pred_bboxes,
                     const int           in_height,
                     const int           in_width,
                     const int           out_height,
                     const int           out_width);
