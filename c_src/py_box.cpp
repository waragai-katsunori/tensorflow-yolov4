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

#include <cmath>
#include <iostream>

/**
 * @param dataset {{x, y, w, h, cls_id}, ...}
 * @param metayolos {{h, w, c, classes, label_smooth_eps, max, iou_thresh,
 *          mask0, ...}}
 * @param anchors {w0, h0, w1, h1, ...} 0 ~ 1
 *
 * @return {Dim(h,w,c+num_masks), ...}
 */
py::list convert_dataset_to_ground_truth(py::array_t<float> &dataset,
                                         py::array_t<float> &metayolos,
                                         py::array_t<float> &anchors) {
    py::list               y_true;
    std::array<float *, 3> gt_one_ptr_v;

    auto   _dataset    = dataset.mutable_unchecked<2>();
    float *dataset_ptr = _dataset.mutable_data(0, 0);

    auto  _metayolos       = metayolos.mutable_unchecked<2>();
    int   num_yolos        = _metayolos.shape(0);
    int   channels         = static_cast<int>(_metayolos(num_yolos - 1, 2));
    int   classes          = static_cast<int>(_metayolos(num_yolos - 1, 3));
    float label_smooth_eps = _metayolos(num_yolos - 1, 4);
    int   num_dataset = fmin(static_cast<int>(_metayolos(num_yolos - 1, 5)),
                           _dataset.shape(0));
    float iou_thresh  = _metayolos(num_yolos - 1, 6);

    int box_size  = 5 + classes;
    int num_masks = channels / box_size;

    std::array<int, 3>                height_v;
    std::array<int, 3>                width_v;
    std::array<std::array<int, 3>, 3> mask_v;
    for(int y = 0; y < num_yolos; y++) {
        height_v[y] = static_cast<int>(_metayolos(y, 0));
        width_v[y]  = static_cast<int>(_metayolos(y, 1));
        for(int n = 0; n < num_masks; n++) {
            mask_v[y][n] = static_cast<int>(_metayolos(y, 7 + n));
        }
    }

    float label_true;
    float label_false;
    if(label_smooth_eps > 0.0) {
        label_true  = 1 - 0.5 * label_smooth_eps;
        label_false = 0.5 * label_smooth_eps;
    } else {
        label_true  = 1;
        label_false = 0;
    }

    auto                 _anchors    = anchors.mutable_unchecked<1>();
    float *              anchors_ptr = _anchors.mutable_data(0);
    int                  num_anchors = _anchors.shape(0) / 2;
    std::array<bool, 15> use_anchor_v;

    for(int y = 0; y < num_yolos; y++) {
        int height = height_v[y];
        int width  = width_v[y];

        py::array_t<float> gt_one({height, width, channels + num_masks});
        y_true.append(gt_one);
        float *gt_one_ptr = gt_one.mutable_data(0, 0, 0);
        gt_one_ptr_v[y]   = gt_one_ptr;

        for(int j = 0; j < height; j++) {
            for(int i = 0; i < width; i++) {
                int stride = (j * width + i) * (channels + 3);
                for(int n = 0; n < num_masks; n++) {
                    int box_index      = n * box_size + stride;
                    int obj_index      = box_index + 4;
                    int cls_index      = box_index + 5;
                    int next_box_index = box_index + box_size;

                    // x, y, w, h
                    gt_one_ptr[box_index]     = 0;
                    gt_one_ptr[box_index + 1] = 0;
                    gt_one_ptr[box_index + 2] = 0;
                    gt_one_ptr[box_index + 3] = 0;
                    // o
                    gt_one_ptr[obj_index] = 0;
                    // c0, c1, ...
                    for(int k = cls_index; k < next_box_index; k++) {
                        gt_one_ptr[k] = label_false;
                    }

                    gt_one_ptr[stride + channels + n] = 0;
                }
            }
        }
    }

    for(int t = 0; t < num_dataset; t++) {
        xywh &truth  = *reinterpret_cast<xywh *>(&dataset_ptr[5 * t]);
        int   cls_id = static_cast<int>(dataset_ptr[5 * t + 4]);
        xywh  truth_shift {.x = 0, .y = 0, .w = truth.w, .h = truth.h};

        // Find best anchor
        float best_iou = 0;
        int   best_n   = 0;
        for(int n = 0; n < num_anchors; n++) {
            xywh  anchor {.x = 0,
                         .y = 0,
                         .w = anchors_ptr[2 * n],
                         .h = anchors_ptr[2 * n + 1]};
            float iou = get_iou(truth_shift, anchor);

            if(iou > best_iou) {
                best_iou = iou;
                best_n   = n;
            }
            // Find over iou_thresh
            use_anchor_v[n] = iou > iou_thresh;
        }
        use_anchor_v[best_n] = true;

        for(int y = 0; y < num_yolos; y++) {
            int    i          = truth.x * width_v[y];
            int    j          = truth.y * height_v[y];
            float *gt_one_ptr = gt_one_ptr_v[y];
            int    stride     = (j * width_v[y] + i) * (channels + 3);

            for(int n = 0; n < num_masks; n++) {
                if(use_anchor_v[mask_v[y][n]]) {
                    int box_index = n * box_size + stride;
                    int obj_index = box_index + 4;
                    int cls_index = box_index + 5;

                    // x, y, w, h
                    gt_one_ptr[box_index] += truth.x;
                    gt_one_ptr[box_index + 1] += truth.y;
                    gt_one_ptr[box_index + 2] += truth.w;
                    gt_one_ptr[box_index + 3] += truth.h;
                    // o
                    gt_one_ptr[obj_index] += 1;
                    // c0, c1, ...
                    gt_one_ptr[cls_index + cls_id] = label_true;

                    gt_one_ptr[stride + channels + n] += 1;
                }
            }
        }
    }

    // Calculate average box
    for(int y = 0; y < num_yolos; y++) {
        int    height     = height_v[y];
        int    width      = width_v[y];
        float *gt_one_ptr = gt_one_ptr_v[y];
        for(int j = 0; j < height; j++) {
            for(int i = 0; i < width; i++) {
                int stride = (j * width + i) * (channels + 3);
                for(int n = 0; n < num_masks; n++) {
                    float num_box = gt_one_ptr[stride + channels + n];
                    if(num_box > 1) {
                        int box_index = n * box_size + stride;
                        int obj_index = box_index + 4;

                        // x, y, w, h
                        gt_one_ptr[box_index] /= num_box;
                        gt_one_ptr[box_index + 1] /= num_box;
                        gt_one_ptr[box_index + 2] /= num_box;
                        gt_one_ptr[box_index + 3] /= num_box;
                        // o
                        gt_one_ptr[obj_index] /= num_box;
                        // one
                        gt_one_ptr[stride + channels + n] = 1;
                    }
                }
            }
        }
    }

    return y_true;
}

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
