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
#include "box.h"

#include <math.h>

float get_diou(const xywh &a,
               const xywh &b,
               const lrtb &a_lrtb,
               const lrtb &b_lrtb,
               const float beta1) {
    lrtb in_lrtb, out_lrtb;
    get_in_out_lrtb(a_lrtb, b_lrtb, in_lrtb, out_lrtb);
    float in_w  = in_lrtb.r - in_lrtb.l;
    float in_h  = in_lrtb.b - in_lrtb.t;
    float out_w = out_lrtb.r - out_lrtb.l;
    float out_h = out_lrtb.b - out_lrtb.t;

    float iou;
    if(in_w <= 0 || in_h <= 0) {
        iou = 0;
        if(out_w == 0 && out_h == 0) { return 0; }
    } else {
        float in  = in_w * in_h;
        float uni = a.w * a.h + b.w * b.h - in;
        iou       = in / uni;
    }

    float c     = out_w * out_w + out_h * out_h;
    float d     = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float rdiou = pow(d / c, beta1);
    return iou - rdiou;
}

void get_in_out_lrtb(const lrtb &a, const lrtb &b, lrtb &in, lrtb &out) {
    if(a.l > b.l) {
        in.l  = a.l;
        out.l = b.l;
    } else {
        in.l  = b.l;
        out.l = a.l;
    }
    if(a.r < b.r) {
        in.r  = a.r;
        out.r = b.r;
    } else {
        in.r  = b.r;
        out.r = a.r;
    }
    if(a.t > b.t) {
        in.t  = a.t;
        out.t = b.t;
    } else {
        in.t  = b.t;
        out.t = a.t;
    }
    if(a.b < b.b) {
        in.b  = a.b;
        out.b = b.b;
    } else {
        in.b  = b.b;
        out.b = a.b;
    }
}

float get_iou(const xywh &a, const xywh &b) {
    lrtb a_lrtb = get_lrtb(a);
    lrtb b_lrtb = get_lrtb(b);
    lrtb in_lrtb;

    in_lrtb.l  = a_lrtb.l > b_lrtb.l ? a_lrtb.l : b_lrtb.l;
    in_lrtb.r  = a_lrtb.r < b_lrtb.r ? a_lrtb.r : b_lrtb.r;
    float in_w = in_lrtb.r - in_lrtb.l;
    if(in_w <= 0) { return 0; }

    in_lrtb.t  = a_lrtb.t > b_lrtb.t ? a_lrtb.t : b_lrtb.t;
    in_lrtb.b  = a_lrtb.b < b_lrtb.b ? a_lrtb.b : b_lrtb.b;
    float in_h = in_lrtb.b - in_lrtb.t;
    if(in_h <= 0) { return 0; }

    float in  = in_w * in_h;
    float uni = a.w * a.h + b.w * b.h - in;
    return in / uni;
}

lrtb get_lrtb(const xywh &a) {
    float w_2 = a.w / 2;
    float h_2 = a.h / 2;
    return lrtb {
        .l = a.x - w_2,
        .r = a.x + w_2,
        .t = a.y - h_2,
        .b = a.y + h_2,
    };
}
