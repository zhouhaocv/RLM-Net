# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import torch

def nms(dets_s,dets_o,scores,labels_s,labels_o, thresh):
    x1_s = dets_s[:, 0]
    y1_s = dets_s[:, 1]
    x2_s = dets_s[:, 2]
    y2_s = dets_s[:, 3]

    x1_o = dets_o[:, 0]
    y1_o = dets_o[:, 1]
    x2_o = dets_o[:, 2]
    y2_o = dets_o[:, 3]

    areas_s = (x2_s - x1_s + 1) * (y2_s - y1_s + 1)
    areas_o = (x2_o - x1_o + 1) * (y2_o - y1_o + 1)

    order = torch.argsort(scores,dim=0,descending=True)

    keep = []
    while len(order) > 0 and len(keep) < 156:
        i = order[0]
        keep.append(i)
        xx1_s = torch.max(x1_s[i], x1_s[order[1:]])
        yy1_s = torch.max(y1_s[i], y1_s[order[1:]])
        xx2_s = torch.min(x2_s[i], x2_s[order[1:]])
        yy2_s = torch.min(y2_s[i], y2_s[order[1:]])

        w_s = torch.clamp(xx2_s - xx1_s + 1,min=0)
        h_s = torch.clamp(yy2_s - yy1_s + 1,min=0)
        inter_s = w_s * h_s
        ovr_s = inter_s / (areas_s[i] + areas_s[order[1:]] - inter_s)

        xx1_o = torch.max(x1_o[i], x1_o[order[1:]])
        yy1_o = torch.max(y1_o[i], y1_o[order[1:]])
        xx2_o = torch.min(x2_o[i], x2_o[order[1:]])
        yy2_o = torch.min(y2_o[i], y2_o[order[1:]])

        w_o = torch.clamp(xx2_o - xx1_o + 1,min=0)
        h_o = torch.clamp(yy2_o - yy1_o + 1,min=0)
        inter_o = w_o * h_o
        ovr_o = inter_o / (areas_o[i] + areas_o[order[1:]] - inter_o)

        ovr = ovr_s * ovr_o
        inds1 = ovr <= thresh

        inds2 = ~((labels_s[order[1:]] == labels_s[i]) & (labels_o[order[1:]] == labels_o[i]))

        inds = inds1 | inds2

        order = order[1:]
        order = order[inds]

    keep = torch.tensor(keep)

    return keep
