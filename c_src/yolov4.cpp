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
#include "py_box.h"
#include "py_layers.h"

namespace py = pybind11;

PYBIND11_MODULE(_common, m) {
    // py_box

    m.def("convert_dataset_to_ground_truth",
          &convert_dataset_to_ground_truth,
          py::arg("dataset").noconvert(),
          py::arg("metayolos").noconvert(),
          py::arg("anchors").noconvert());

    m.def("get_yolo_detections",
          &get_yolo_detections,
          py::arg("yolo_0").noconvert(),
          py::arg("yolo_1").noconvert(),
          py::arg("yolo_2").noconvert(),
          py::arg("mask_0").noconvert(),
          py::arg("mask_1").noconvert(),
          py::arg("mask_2").noconvert(),
          py::arg("anchors").noconvert(),
          py::arg("beta_nms"),
          py::arg("new_coords"));

    m.def("get_yolo_tiny_detections",
          &get_yolo_tiny_detections,
          py::arg("yolo_0").noconvert(),
          py::arg("yolo_1").noconvert(),
          py::arg("mask_0").noconvert(),
          py::arg("mask_1").noconvert(),
          py::arg("anchors").noconvert(),
          py::arg("beta_nms"),
          py::arg("new_coords"));

    m.def("fit_to_original",
          &fit_to_original,
          py::arg("pred_bboxes").noconvert(),
          py::arg("in_height"),
          py::arg("in_width"),
          py::arg("out_height"),
          py::arg("out_width"));

    // py_layers

    m.def("yolo_tpu_layer",
          &yolo_tpu_layer,
          py::arg("x").noconvert(),
          py::arg("logi").noconvert(),
          py::arg("num_masks"),
          py::arg("scale_x_y"));

    m.def("yolo_tpu_layer_new_coords",
          &yolo_tpu_layer_new_coords,
          py::arg("logi").noconvert(),
          py::arg("num_masks"),
          py::arg("scale_x_y"));
}
