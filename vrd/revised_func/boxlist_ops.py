# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.layers import nms as _box_nms


def proposal_matcher(match_quality_matrix,high_threshold=0.45, low_threshold=0.25):
    """
    Args:
        match_quality_matrix (Tensor[float]): an MxN tensor, containing the
        pairwise quality between M ground-truth elements and N predicted elements.

    Returns:
        matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
        [0, M - 1] or a negative value indicating that prediction i could not
        be matched.
    """
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2
    assert low_threshold <= high_threshold
    if match_quality_matrix.numel() == 0:
        # empty targets or proposals not supported during training
        if match_quality_matrix.shape[0] == 0:
            raise ValueError(
                "No ground-truth boxes available for one of the images "
                "during training")
        else:
            raise ValueError(
                "No proposal boxes available for one of the images "
                "during training")

    # match_quality_matrix is M (gt) x N (predicted)
    # Max over gt elements (dim 0) to find best gt candidate for each prediction
    matched_vals, matches = match_quality_matrix.max(dim=0)

    # Assign candidate matches with low quality to negative (unassigned) values
    below_low_threshold = matched_vals < low_threshold
    between_thresholds = (matched_vals >= low_threshold) & (
        matched_vals < high_threshold
    )
    matches[below_low_threshold] = BELOW_LOW_THRESHOLD
    matches[between_thresholds] = BETWEEN_THRESHOLDS

    return matches


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="score"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def compute_objectpairs_iou(targets, proposals):
    match_quality_matrix = boxlist_iou(targets, proposals)
    objects_pairs = targets.get_field('objects_pairs')
    match_s = match_quality_matrix[objects_pairs[:,0],:].permute(1,0)
    match_o = match_quality_matrix[objects_pairs[:,1],:].permute(1,0)

    match_s = match_s.unsqueeze(1).expand(match_s.size(0),match_s.size(0),match_s.size(1))
    match_o = match_o.unsqueeze(0).expand(match_o.size(0),match_o.size(0),match_o.size(1))
    match = match_s*match_o
    match = match.view(match_s.size(0)**2,match_s.size(2)).permute(1,0)

    return match
