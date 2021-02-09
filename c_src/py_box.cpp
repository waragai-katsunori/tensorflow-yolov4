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

#include "box.h"

#include <iostream>

void fit_to_original(py::array_t<float> &pred_bboxes,
                     const int           in_height,
                     const int           in_width,
                     const int           out_height,
                     const int           out_width) {
    auto      preds     = pred_bboxes.mutable_unchecked<2>();
    float *   preds_ptr = pred_bboxes.mutable_data(0, 0);
    const int num_preds = preds.shape(0);
    float     out_w_h   = out_width / static_cast<float>(out_height);
    float     in_w_h    = in_width / static_cast<float>(in_height);
    float     scale     = out_w_h / in_w_h;

    if(scale > 1.03) {
        for(int i = 0; i < num_preds; i++) {
            int stride            = i * 9;
            preds_ptr[stride + 1] = scale * (preds_ptr[stride + 1] - 0.5) + 0.5;
            preds_ptr[stride + 3] = scale * preds_ptr[stride + 3];
        }
    } else if(scale < 0.97) {
        scale = 1 / scale;
        for(int i = 0; i < num_preds; i++) {
            int stride            = i * 9;
            preds_ptr[stride + 0] = scale * (preds_ptr[stride + 0] - 0.5) + 0.5;
            preds_ptr[stride + 2] = scale * preds_ptr[stride + 2];
        }
    }
}

py::array_t<float> yolo_diou_nms(py::array_t<float> &candidates, float beta1) {
    const float thresh = .005;    // Pr(obj) * IOU > 0.05% and Pr(cls) > 0.05%
    const float nms_thresh = .5;

    // sum (height_i * width_i * anchors_i), (x, y, w, h, o, c0, ...)
    auto                 cands     = candidates.mutable_unchecked<2>();
    float *              cands_ptr = cands.mutable_data(0, 0);
    const int            num_cands = cands.shape(0);
    const int            stride    = cands.shape(1);
    const int            classes   = stride - 5;
    int                  count     = 0;
    std::vector<float *> v_ptr;


    // find obj > thresh
    // set c0 = obj * c0 > thresh ? obj * c0 : 0
    for(py::ssize_t i = 0; i < num_cands; i++) {
        int c_box_index = count * stride;
        int c_obj_index = c_box_index + 4;
        int c_cls_index = c_box_index + 5;
        int i_box_index = i * stride;
        int i_obj_index = i_box_index + 4;
        int i_cls_index = i_box_index + 5;

        float obj = cands_ptr[i_obj_index];
        if(obj > thresh) {
            for(int j = 0; j < 4; j++) {
                // copy x, y, w, h
                cands_ptr[c_box_index + j] = cands_ptr[i_box_index + j];
            }
            // copy o
            cands_ptr[c_obj_index] = obj;

            bool exist = false;
            for(int j = 0; j < classes; j++) {
                float prob = obj * cands_ptr[i_cls_index + j];
                if(prob > thresh) {
                    exist                      = true;
                    cands_ptr[c_cls_index + j] = prob;
                } else {
                    cands_ptr[c_cls_index + j] = 0;
                }
            }
            if(exist) { count++; }
        }
    }

    v_ptr.reserve(count);

    for(int i = 0; i < count; i++) { v_ptr.push_back(&cands_ptr[stride * i]); }

    for(int k = 0; k < classes; k++) {
        // Descending
        std::sort(v_ptr.begin(), v_ptr.end(), [k](float_t *a, float *b) {
            return a[5 + k] > b[5 + k];
        });

        std::vector<float *>::iterator iter = v_ptr.begin();
        for(float *a: v_ptr) {
            iter++;    // next(a)
            if(a[5 + k] == 0) continue;

            xywh &a_xywh = *reinterpret_cast<xywh *>(a);
            lrtb  a_lrtb = get_lrtb(a_xywh);
            for(auto it = iter; it != v_ptr.end(); it++) {
                float *b      = *it;
                xywh & b_xywh = *reinterpret_cast<xywh *>(b);
                lrtb   b_lrtb = get_lrtb(b_xywh);
                float  diou   = get_diou(a_xywh, b_xywh, a_lrtb, b_lrtb, beta1);
                // remove
                if(diou > nms_thresh) { b[5 + k] = 0; }
            }
        }
    }

    // pred_box
    // x, y, w, h, o, class_id, p(c), class_id, p(c)
    auto   result  = py::array_t<float>({count, 9});
    float *ret_ptr = result.mutable_data(0, 0);
    for(int i = 0; i < count; i++) {
        // set x, y, w, h, o
        int ret_stride = i * 9;
        for(int k = 0; k < 5; k++) { ret_ptr[ret_stride + k] = v_ptr[i][k]; }
        // init
        for(int k = 5; k < 9; k++) { ret_ptr[ret_stride + k] = 0; }

        // find
        for(int k = 0; k < classes; k++) {
            float prob = v_ptr[i][5 + k];
            if(prob == 0) continue;
            if(prob > ret_ptr[ret_stride + 6]) {
                ret_ptr[ret_stride + 5] = k;
                ret_ptr[ret_stride + 6] = prob;
            } else if(prob > ret_ptr[ret_stride + 8]) {
                ret_ptr[ret_stride + 7] = k;
                ret_ptr[ret_stride + 8] = prob;
            }
        }
    }

    return result;
}
