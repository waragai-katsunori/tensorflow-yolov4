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
 * @param logi logistic(x), this is result.
 * @param num_masks
 * @param scale_x_y
 */
void yolo_tpu_layer(py::array_t<float> &x,
                    py::array_t<float> &logi,
                    const int           num_masks,
                    const float         scale_x_y) {
    auto         _x    = x.unchecked<4>();
    const float *x_ptr = _x.data(0, 0, 0, 0);

    auto   _logi    = logi.mutable_unchecked<4>();
    float *logi_ptr = _logi.mutable_data(0, 0, 0, 0);
    float *yolo_ptr = logi_ptr;

    const int height   = _x.shape(1);
    const int width    = _x.shape(2);
    const int channels = _x.shape(3);
    const int box_size = channels / num_masks;

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int stride = (y * width + x) * channels;
            for(int n = 0; n < num_masks; n++) {
                int box_index = stride + n * box_size;

                // x, y
                if(scale_x_y > 1.001f) {
                    yolo_ptr[box_index] = scale_x_y * logi_ptr[box_index]
                                          - (0.5 * (scale_x_y - 1));
                    yolo_ptr[box_index + 1]
                        = scale_x_y * logi_ptr[box_index + 1]
                          - (0.5 * (scale_x_y - 1));
                }

                // w, h
                yolo_ptr[box_index + 2] = x_ptr[box_index + 2];
                yolo_ptr[box_index + 3] = x_ptr[box_index + 3];
            }
        }
    }
}

/**
 * @param logi logistic(x), Dim(1, height, width, channels), this is result.
 * @param num_masks
 * @param scale_x_y
 */
void yolo_tpu_layer_new_coords(py::array_t<float> &logi,
                               const int           num_masks,
                               const float         scale_x_y) {
    auto   _logi    = logi.mutable_unchecked<4>();
    float *logi_ptr = _logi.mutable_data(0, 0, 0, 0);
    float *yolo_ptr = logi_ptr;

    const int height   = _logi.shape(1);
    const int width    = _logi.shape(2);
    const int channels = _logi.shape(3);
    const int box_size = channels / num_masks;

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int stride = (y * width + x) * channels;
            for(int n = 0; n < num_masks; n++) {
                int box_index = stride + n * box_size;

                // x, y
                yolo_ptr[box_index]
                    = scale_x_y * logi_ptr[box_index] - (0.5 * (scale_x_y - 1));
                yolo_ptr[box_index + 1] = scale_x_y * logi_ptr[box_index + 1]
                                          - (0.5 * (scale_x_y - 1));
            }
        }
    }
}
