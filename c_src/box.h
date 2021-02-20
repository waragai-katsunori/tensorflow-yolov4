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

typedef struct xywh {
    // center x, center y, width, height
    float x, y, w, h;
} xywh;

typedef struct lrtb {
    // top, left, bottom, right
    float l, r, t, b;
} lrtb;

typedef struct pred_xywh {
    // center x, center y, width, height, cls_id, prob
    float x, y, w, h, cls_id, prob;
} pred_xywh;

float get_diou(const xywh &a_xywh,
               const xywh &b_xywh,
               const lrtb &a_lrtb,
               const lrtb &b_lrtb,
               const float beta1);

void get_in_out_lrtb(const lrtb &a, const lrtb &b, lrtb &in, lrtb &out);

float get_iou(const xywh &a, const xywh &b);

lrtb get_lrtb(const xywh &a);