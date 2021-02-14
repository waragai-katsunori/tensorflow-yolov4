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
#include "py_layers.h"

#include <cmath>

/**
 * @param x Dim(1, height, width, channels)
 * @param logi logistic(x)
 * @param anchors {{w, h}, ...} 0 ~ 1
 *
 * @return Dim(1, height, width, channels)
 */
py::array_t<float> yolo_layer(py::array_t<float> &x,
                              py::array_t<float> &logi,
                              py::array_t<float> &anchors,
                              const float         scale_x_y) {
    auto         _x    = x.unchecked<4>();
    const float *x_ptr = _x.data(0, 0, 0, 0);

    auto         _logi    = logi.unchecked<4>();
    const float *logi_ptr = _logi.data(0, 0, 0, 0);

    auto         _anchors = anchors.unchecked<2>();
    const float *anch_ptr = _anchors.data(0, 0);

    int height     = _x.shape(1);
    int width      = _x.shape(2);
    int channels   = _x.shape(3);
    int num_anchor = _anchors.shape(0);
    int box_size   = channels / num_anchor;

    py::array_t<float> yolo({1, height, width, channels});
    float *            yolo_ptr = yolo.mutable_data(0, 0, 0, 0);

    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            int stride = (j * width + i) * channels;
            for(int n = 0; n < num_anchor; n++) {
                int box_index      = stride + n * box_size;
                int obj_index      = box_index + 4;
                int next_box_index = box_index + box_size;
                int anchor_index   = n * 2;

                // x, y
                if(scale_x_y > 1.001f) {
                    yolo_ptr[box_index]
                        = ((logi_ptr[box_index] - 0.5) * scale_x_y + 0.5 + i)
                          / static_cast<float>(width);
                    yolo_ptr[box_index + 1]
                        = ((logi_ptr[box_index + 1] - 0.5) * scale_x_y + 0.5
                           + j)
                          / static_cast<float>(height);
                } else {
                    yolo_ptr[box_index]
                        = (logi_ptr[box_index] + i) / static_cast<float>(width);
                    yolo_ptr[box_index + 1] = (logi_ptr[box_index + 1] + j)
                                              / static_cast<float>(height);
                }

                // w, h
                yolo_ptr[box_index + 2]
                    = exp(x_ptr[box_index + 2]) * anch_ptr[anchor_index];
                yolo_ptr[box_index + 3]
                    = exp(x_ptr[box_index + 3]) * anch_ptr[anchor_index + 1];

                for(int k = obj_index; k < next_box_index; k++) {
                    yolo_ptr[k] = logi_ptr[k];
                }
            }
        }
    }

    return yolo;
}