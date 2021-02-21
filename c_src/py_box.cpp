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
#include <list>

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

/**
 * @param pred_bboxes Dim(candidates after NMS, pred_xywh)
 * @param in_height input data height
 * @param in_width input data width
 * @param out_height original image height
 * @param out_width original image width
 */
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
            int stride            = i * 6;
            preds_ptr[stride + 1] = scale * (preds_ptr[stride + 1] - 0.5) + 0.5;
            preds_ptr[stride + 3] = scale * preds_ptr[stride + 3];
        }
    } else if(scale < 0.97) {
        scale = 1 / scale;
        for(int i = 0; i < num_preds; i++) {
            int stride            = i * 6;
            preds_ptr[stride + 0] = scale * (preds_ptr[stride + 0] - 0.5) + 0.5;
            preds_ptr[stride + 2] = scale * preds_ptr[stride + 2];
        }
    }
}


/**
 * @param candidates Dim(None, 5 + classes)
 * @param bbox_size 5 + classes
 * @param beta_nms
 *
 * @return Dim(candidates after NMS, pred_xywh)
 */
py::array_t<float>
    diou_nms(std::list<float *> &candidates, int bbox_size, float beta_nms) {
    const float nms_thresh = .6;

    if(candidates.size() == 0) {
        py::array_t<float> result({1, 6});
        for(int i = 0; i < 6; i++) { result.mutable_at(0, i) = 0; }
        return result;
    }

    // DIoU NMS
    if(candidates.size() != 1) {
        for(int c = 5; c < bbox_size; c++) {
            // Descending
            candidates.sort([c](float *a, float *b) { return a[c] > b[c]; });

            std::list<float *>::iterator iter = candidates.begin();
            for(float *a: candidates) {
                iter++;    // next(a)
                if(a[c] == 0) continue;

                xywh &a_xywh = *reinterpret_cast<xywh *>(a);
                lrtb  a_lrtb = get_lrtb(a_xywh);
                for(auto it = iter; it != candidates.end(); it++) {
                    float *b      = *it;
                    xywh & b_xywh = *reinterpret_cast<xywh *>(b);
                    lrtb   b_lrtb = get_lrtb(b_xywh);
                    float  diou
                        = get_diou(a_xywh, b_xywh, a_lrtb, b_lrtb, beta_nms);
                    // remove
                    if(diou > nms_thresh) { b[c] = 0; }
                }
            }
        }
    }

    // pred_box
    // x, y, w, h, class_id, p(c)
    for(auto it = candidates.begin(); it != candidates.end();) {
        // find
        float *bbox = *it;
        bbox[4]     = 0;    // class_id
        for(int c = 6; c < bbox_size; c++) {
            if(bbox[c] == 0) continue;
            if(bbox[c] > bbox[5]) {
                bbox[4] = c - 5;
                bbox[5] = bbox[c];
            }
        }

        if(bbox[5] < 0.001) {
            it = candidates.erase(it);
        } else {
            it++;
        }
    }

    if(candidates.size() == 0) {
        py::array_t<float> result({1, 6});
        for(int i = 0; i < 6; i++) { result.mutable_at(0, i) = 0; }
        return result;
    }

    int                _num = candidates.size();
    py::array_t<float> result({_num, 6});
    int                i = 0;
    // copy
    for(auto it = candidates.begin(); it != candidates.end(); it++) {
        float *bbox  = *it;
        float *rbbox = result.mutable_data(i++, 0);
        for(int j = 0; j < 6; j++) { rbbox[j] = bbox[j]; }
    }

    return result;
}

/**
 * @param yolo Dim(1, height, widht, bbox * len(mask))
 * @param mask
 * @param anchors Dim(None, 2): 0 ~ 1
 * @param beta_nms
 * @param new_coords
 * @param prob_thresh
 *
 * @return Dim(candidates after NMS, pred_xywh)
 */
py::array_t<float> get_yolo_detections(py::array_t<float> &yolo_0,
                                       py::array_t<float> &yolo_1,
                                       py::array_t<float> &yolo_2,
                                       py::array_t<int> &  mask_0,
                                       py::array_t<int> &  mask_1,
                                       py::array_t<int> &  mask_2,
                                       py::array_t<float> &anchors,
                                       float               beta_nms,
                                       bool                new_coords,
                                       float               prob_thresh) {
    std::array<py::detail::unchecked_mutable_reference<float, 4>, 3> yolos = {
        yolo_0.mutable_unchecked<4>(),
        yolo_1.mutable_unchecked<4>(),
        yolo_2.mutable_unchecked<4>(),
    };

    std::array<py::detail::unchecked_mutable_reference<int, 1>, 3> masks {
        mask_0.mutable_unchecked<1>(),
        mask_1.mutable_unchecked<1>(),
        mask_2.mutable_unchecked<1>(),
    };

    auto biases = anchors.mutable_unchecked<2>();

    const int          num_bbox  = masks[0].shape(0);
    const int          bbox_size = yolos[0].shape(3) / num_bbox;
    std::list<float *> candidates_l;

    for(int i = 0; i < 3; i++) {
        int height = yolos[i].shape(1);
        int width  = yolos[i].shape(2);
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                for(int b = 0; b < num_bbox; b++) {
                    float *bbox = yolos[i].mutable_data(0, y, x, b * bbox_size);
                    float  obj  = bbox[4];
                    bool   exist = false;
                    int    mask  = masks[i](b);

                    if(obj > prob_thresh) {
                        for(int c = 5; c < bbox_size; c++) {
                            float prob = bbox[c] * obj;
                            if(prob > prob_thresh) {
                                bbox[c] = prob;
                                exist   = true;
                            } else {
                                bbox[c] = 0;
                            }
                        }
                    }

                    if(exist) {
                        bbox[0] = (bbox[0] + x) / width;
                        bbox[1] = (bbox[1] + y) / height;
                        if(new_coords) {
                            bbox[2] = bbox[2] * bbox[2] * 4 * biases(mask, 0);
                            bbox[3] = bbox[3] * bbox[3] * 4 * biases(mask, 1);
                        } else {
                            bbox[2] = exp(bbox[2]) * biases(mask, 0);
                            bbox[3] = exp(bbox[3]) * biases(mask, 1);
                        }
                        candidates_l.push_back(bbox);
                    }
                }
            }
        }
    }

    return diou_nms(candidates_l, bbox_size, beta_nms);
}

/**
 * @param yolo Dim(1, height, widht, bbox * len(mask))
 * @param mask
 * @param anchors Dim(None, 2): 0 ~ 1
 * @param beta_nms
 * @param new_coords
 * @param prob_thresh
 *
 * @return Dim(candidates after NMS, pred_xywh)
 */
py::array_t<float> get_yolo_tiny_detections(py::array_t<float> &yolo_0,
                                            py::array_t<float> &yolo_1,
                                            py::array_t<int> &  mask_0,
                                            py::array_t<int> &  mask_1,
                                            py::array_t<float> &anchors,
                                            float               beta_nms,
                                            bool                new_coords,
                                            float               prob_thresh) {
    std::array<py::detail::unchecked_mutable_reference<float, 4>, 2> yolos = {
        yolo_0.mutable_unchecked<4>(),
        yolo_1.mutable_unchecked<4>(),
    };

    std::array<py::detail::unchecked_mutable_reference<int, 1>, 2> masks {
        mask_0.mutable_unchecked<1>(),
        mask_1.mutable_unchecked<1>(),
    };

    auto biases = anchors.mutable_unchecked<2>();

    const int          num_bbox  = masks[0].shape(0);
    const int          bbox_size = yolos[0].shape(3) / num_bbox;
    std::list<float *> candidates_l;

    for(int i = 0; i < 2; i++) {
        int height = yolos[i].shape(1);
        int width  = yolos[i].shape(2);
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                for(int b = 0; b < num_bbox; b++) {
                    float *bbox = yolos[i].mutable_data(0, y, x, b * bbox_size);
                    float  obj  = bbox[4];
                    bool   exist = false;
                    int    mask  = masks[i](b);

                    if(obj > prob_thresh) {
                        for(int c = 5; c < bbox_size; c++) {
                            float prob = bbox[c] * obj;
                            if(prob > prob_thresh) {
                                bbox[c] = prob;
                                exist   = true;
                            } else {
                                bbox[c] = 0;
                            }
                        }
                    }

                    if(exist) {
                        bbox[0] = (bbox[0] + x) / width;
                        bbox[1] = (bbox[1] + y) / height;
                        if(new_coords) {
                            bbox[2] = bbox[2] * bbox[2] * 4 * biases(mask, 0);
                            bbox[3] = bbox[3] * bbox[3] * 4 * biases(mask, 1);
                        } else {
                            bbox[2] = exp(bbox[2]) * biases(mask, 0);
                            bbox[3] = exp(bbox[3]) * biases(mask, 1);
                        }
                        candidates_l.push_back(bbox);
                    }
                }
            }
        }
    }

    return diou_nms(candidates_l, bbox_size, beta_nms);
}
