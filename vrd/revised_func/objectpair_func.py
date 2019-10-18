import torch
from .boxlist_ops import compute_objectpairs_iou,proposal_matcher
from maskrcnn_benchmark.structures.bounding_box import BoxList
from .nms import nms
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

def compute_spatial_objectpairs(target):
    W,H = target.size
    W = float(W)
    H = float(H)
    num_boxes = len(target)
    boundingboxes = target.bbox
    eps = torch.tensor(10e-16,device = boundingboxes.device)

    object_boundingboxes = boundingboxes.repeat(num_boxes,1)
    subject_boundingboxes = boundingboxes.repeat(1,num_boxes).view(-1,4).contiguous()

    xs_o = object_boundingboxes[:,0].view(-1,1)
    xm_o = object_boundingboxes[:,2].view(-1,1)
    ys_o = object_boundingboxes[:,1].view(-1,1)
    ym_o = object_boundingboxes[:,3].view(-1,1)
    xs_s = subject_boundingboxes[:,0].view(-1,1)
    xm_s = subject_boundingboxes[:,2].view(-1,1)
    ys_s = subject_boundingboxes[:,1].view(-1,1)
    ym_s = subject_boundingboxes[:,3].view(-1,1)

    x_0 = xs_s/W
    x_1 = ys_s/H
    x_2 = xm_s/W
    x_3 = ym_s/H
    x_4 = ((ym_s-ys_s)*(xm_s-xs_s))/(W*H)
    x_5 = xs_o/W
    x_6 = ys_o/H
    x_7 = xm_o/W
    x_8 = ym_o/H
    x_9 = ((ym_o-ys_o)*(xm_o-xs_o))/(W*H)

    w_o = torch.max(xm_o-xs_o,eps)
    h_o = torch.max(ym_o-ys_o,eps)
    w_s = torch.max(xm_s-xs_s,eps)
    h_s = torch.max(ym_s-ys_s,eps)
    x_10 = (xs_s-xs_o)/w_o
    x_11 = (ys_s-ys_o)/h_o
    x_12 = torch.log(w_s/w_o)
    x_13 = torch.log(h_s/h_o)
    spatial_info=torch.cat((x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13),dim=1)

    spatial_info = torch.div(spatial_info,torch.norm(spatial_info,dim=1).unsqueeze(-1)).float()

    return spatial_info
    
def compute_labels(targets):
    objects_pairs = targets.get_field('objects_pairs')
    num_boxes = len(targets)
    labels = torch.zeros(num_boxes**2)
    for pair in objects_pairs:
        idx = pair[0]*num_boxes+pair[1]
        labels[idx] = 1
    mask = (torch.ones((num_boxes,num_boxes))-torch.eye(num_boxes)).view(1,-1).squeeze(0)

    return labels,mask

def compute_proposals_labels(proposals,targets):
    objects_pairs = targets.get_field('objects_pairs')
    num_boxes = len(proposals)
    match_matrix = compute_objectpairs_iou(targets, proposals)
    matched_idxs = proposal_matcher(match_matrix)

    labels = torch.clamp(matched_idxs+1,0.0,1.0).float()
    mask = (torch.ones((num_boxes,num_boxes))-torch.eye(num_boxes)).view(1,-1).squeeze(0)
    ignore_idx = matched_idxs==-2
    mask[ignore_idx] = 0

    return labels,mask

def BalancedPN(x_proposals,x_targets,labels_per_image_proposals,labels_per_image_targets,devices):
    """
    """
    if len(x_proposals) !=0:
        batch_size_per_image = 256
        positive_fraction = 0.25
        targets_num = x_targets.size(0)

        positive_proposals = torch.nonzero(labels_per_image_proposals == 1).squeeze(1)
        negative_proposals = torch.nonzero(labels_per_image_proposals == 0).squeeze(1)

        positive_targets = torch.nonzero(labels_per_image_targets == 1).squeeze(1)
        negative_targets = torch.nonzero(labels_per_image_targets == 0).squeeze(1)

        num_pos = int(batch_size_per_image * positive_fraction)
        num_pos = max(min(positive_proposals.numel(), num_pos-positive_targets.numel()),0)

        num_neg = max(batch_size_per_image-targets_num- num_pos,0)
        num_neg = min(negative_proposals.numel(), num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive_proposals.numel(), device=positive_proposals.device)[:num_pos]
        perm2 = torch.randperm(negative_proposals.numel(), device=negative_proposals.device)[:num_neg]

        pos_idx_per_image = positive_proposals[perm1]
        neg_idx_per_image = negative_proposals[perm2]

        # create binary mask from indices
        pos_idx_per_image_mask = torch.zeros_like(
            labels_per_image_proposals, dtype=torch.uint8
        )
        neg_idx_per_image_mask = torch.zeros_like(
            labels_per_image_proposals, dtype=torch.uint8
        )
        pos_idx_per_image_mask[pos_idx_per_image] = 1
        neg_idx_per_image_mask[neg_idx_per_image] = 1

        img_sampled_inds = torch.nonzero(pos_idx_per_image_mask | neg_idx_per_image_mask).squeeze(1)
        labels_per_image_proposals = labels_per_image_proposals[img_sampled_inds]
        x_proposals = x_proposals[img_sampled_inds]

        labels_per_image_targets = labels_per_image_targets.to(devices)

        x=torch.cat((x_targets,x_proposals),0)
        labels_per_image = torch.cat((labels_per_image_targets,labels_per_image_proposals),0)
    else:
        labels_per_image_targets = labels_per_image_targets.to(devices)
        return x_targets,labels_per_image_targets


    return x, labels_per_image

def gen_objectpair_targets(targets):
    objectpair_targets = []
    for i,target1 in enumerate(targets):
        target = target1.copy_with_fields(['ids','objects_pairs','predicate_label','labels'])

        objects_pairs = target.get_field('objects_pairs')
        predicate_label= target.get_field('predicate_label')
        labels= target.get_field('labels')
        bounding_box = target.bbox
        img_size = target.size
        if len(objects_pairs) !=0:

            subject_boundingboxes = bounding_box[objects_pairs[:,0],:]
            object_boundingboxes = bounding_box[objects_pairs[:,1],:]
            subject_category = labels[objects_pairs[:,0]]
            object_category = labels[objects_pairs[:,1]]

            xs = torch.min(subject_boundingboxes[:,0],object_boundingboxes[:,0]).view(-1,1)
            ys = torch.min(subject_boundingboxes[:,1],object_boundingboxes[:,1]).view(-1,1)
            xm = torch.max(subject_boundingboxes[:,2],object_boundingboxes[:,2]).view(-1,1)
            ym = torch.max(subject_boundingboxes[:,3],object_boundingboxes[:,3]).view(-1,1)
            boxes = torch.cat((xs,ys,xm,ym),1)

            objectpair_target = BoxList(boxes, img_size, mode="xyxy")
            objectpair_target.add_field("label", predicate_label)
            objectpair_target.add_field("subject_boundingboxes", subject_boundingboxes)
            objectpair_target.add_field("object_boundingboxes", object_boundingboxes)
            objectpair_target.add_field("subject_category", subject_category)
            objectpair_target.add_field("object_category", object_category)
        else:
            objectpair_target = BoxList(torch.tensor([],device=bounding_box.device).view(-1,4), img_size, mode="xyxy")
            objectpair_target.add_field("label", predicate_label.new_empty((0)))
            objectpair_target.add_field("subject_boundingboxes", torch.tensor([],device=bounding_box.device).view(-1,4))
            objectpair_target.add_field("object_boundingboxes", torch.tensor([],device=bounding_box.device).view(-1,4))
            objectpair_target.add_field("subject_category", predicate_label.new_empty((0)))
            objectpair_target.add_field("object_category", predicate_label.new_empty((0)))

        objectpair_targets.append(objectpair_target)

    return objectpair_targets

def gen_objectpair_proposals(proposals,targets):
    objectpair_proposals = []
    for i,(proposals1,targets1) in enumerate(zip(proposals,targets)):

        objects_pairs = targets1.get_field('objects_pairs')
        predicate_tar_label = targets1.get_field('predicate_label')

        proposal = proposals1.copy_with_fields(['labels','scores'])
        labels_pro= proposal.get_field('labels')
        bounding_box = proposal.bbox
        img_size = proposal.size

        num_boxes = len(proposals1)
        if num_boxes !=0:
            match_matrix = compute_objectpairs_iou(targets1, proposals1)
            matched_idxs = proposal_matcher(match_matrix,high_threshold=0.45)
            positive_proposals = torch.nonzero(matched_idxs >= 0).squeeze(1)

            a=torch.linspace(0,num_boxes-1,num_boxes).long()
            objectpairs_pro_idx = torch.cat((a.repeat(a.size(0),1).permute(1,0).contiguous().view(-1,1),a.repeat(1,a.size(0)).permute(1,0).view(-1,1)),1)
            
            objectpairs_pro_idx = objectpairs_pro_idx[positive_proposals]
            predicate_pro_label = predicate_tar_label[matched_idxs[positive_proposals]]

            # to find the objectpairs' categories corresponding to targets
            objectpairs_tar_idx = objects_pairs[matched_idxs[positive_proposals]]
            labels_tar = targets1.get_field('labels')
            if len(objectpairs_pro_idx) !=0:
                subject_pro_category = labels_pro[objectpairs_pro_idx[:,0]]
                object_pro_category =  labels_pro[objectpairs_pro_idx[:,1]]
                subject_tar_category = labels_tar[objectpairs_tar_idx[:,0]]
                object_tar_category =  labels_tar[objectpairs_tar_idx[:,1]]
                idx = (object_tar_category==object_pro_category)&(subject_tar_category==subject_pro_category)
                objectpairs_pro_idx = objectpairs_pro_idx[idx]
                predicate_pro_label = predicate_pro_label[idx]

            if len(objectpairs_pro_idx) !=0:

                subject_category = labels_pro[objectpairs_pro_idx[:,0]]
                object_category =  labels_pro[objectpairs_pro_idx[:,1]]

                subject_boundingboxes = bounding_box[objectpairs_pro_idx[:,0],:]
                object_boundingboxes = bounding_box[objectpairs_pro_idx[:,1],:]

                xs = torch.min(subject_boundingboxes[:,0],object_boundingboxes[:,0]).view(-1,1)
                ys = torch.min(subject_boundingboxes[:,1],object_boundingboxes[:,1]).view(-1,1)
                xm = torch.max(subject_boundingboxes[:,2],object_boundingboxes[:,2]).view(-1,1)
                ym = torch.max(subject_boundingboxes[:,3],object_boundingboxes[:,3]).view(-1,1)
                boxes = torch.cat((xs,ys,xm,ym),1)

                objectpair_proposal = BoxList(boxes, img_size, mode="xyxy")
                objectpair_proposal.add_field("label", predicate_pro_label)
                objectpair_proposal.add_field("subject_boundingboxes", subject_boundingboxes)
                objectpair_proposal.add_field("object_boundingboxes", object_boundingboxes)
                objectpair_proposal.add_field("subject_category", subject_category)
                objectpair_proposal.add_field("object_category", object_category)

                objectpair_proposals.append(objectpair_proposal)

                continue

        objectpair_proposal = BoxList(torch.tensor([],device=bounding_box.device).view(-1,4), img_size, mode="xyxy")
        objectpair_proposal.add_field("label", labels_pro.new_empty((0)))
        objectpair_proposal.add_field("subject_boundingboxes", torch.tensor([],device=bounding_box.device).view(-1,4))
        objectpair_proposal.add_field("object_boundingboxes", torch.tensor([],device=bounding_box.device).view(-1,4))
        objectpair_proposal.add_field("subject_category", labels_pro.new_empty((0)))
        objectpair_proposal.add_field("object_category", labels_pro.new_empty((0)))
        objectpair_proposals.append(objectpair_proposal)

    return objectpair_proposals

def combineboxlist(targets,proposals):
    objectpairs = []
    predicate_labels = []
    for i,(proposals1,targets1) in enumerate(zip(proposals,targets)):
        bounding_box_pro = proposals1.bbox
        bounding_box_tar = targets1.bbox
        boxes = torch.cat((bounding_box_pro,bounding_box_tar),0)
        img_size = targets1.size

        predicate_label_pro = proposals1.get_field('label')
        predicate_label_tar = targets1.get_field('label')
        predicate_label = torch.cat((predicate_label_pro,predicate_label_tar),0)

        subject_boundingboxes_pro = proposals1.get_field('subject_boundingboxes')
        subject_boundingboxes_tar = targets1.get_field('subject_boundingboxes')
        subject_boundingboxes = torch.cat((subject_boundingboxes_pro,subject_boundingboxes_tar),0)

        object_boundingboxes_pro = proposals1.get_field('object_boundingboxes')
        object_boundingboxes_tar = targets1.get_field('object_boundingboxes')
        object_boundingboxes = torch.cat((object_boundingboxes_pro,object_boundingboxes_tar),0)

        subject_category_pro = proposals1.get_field('subject_category')
        subject_category_tar = targets1.get_field('subject_category')
        subject_category = torch.cat((subject_category_pro,subject_category_tar),0)

        object_category_pro = proposals1.get_field('object_category')
        object_category_tar = targets1.get_field('object_category')
        object_category = torch.cat((object_category_pro,object_category_tar),0)

        objectpair = BoxList(boxes, img_size, mode="xyxy")
        objectpair.add_field("label", predicate_label)
        objectpair.add_field("subject_boundingboxes", subject_boundingboxes)
        objectpair.add_field("object_boundingboxes", object_boundingboxes)
        objectpair.add_field("subject_category", subject_category)
        objectpair.add_field("object_category", object_category)

        objectpairs.append(objectpair)
        predicate_labels.append(predicate_label)

    return objectpairs, predicate_labels

def gen_objectpairs(proposals,filter_scores):
    objectpairs_list = []

    labels= proposals.get_field('labels')
    object_scores = proposals.get_field('scores')
    bounding_box = proposals.bbox
    img_size = proposals.size
    num_boxes = len(proposals)

    if num_boxes !=0:

        a=torch.linspace(0,num_boxes-1,num_boxes).long()
        objectpairs_idx = torch.cat((a.repeat(a.size(0),1).permute(1,0).contiguous().view(-1,1),a.repeat(1,a.size(0)).permute(1,0).view(-1,1)),1)

        detection_scores = object_scores.repeat(a.size(0),1).permute(1,0).contiguous().view(-1,1)*object_scores.repeat(1,a.size(0)).permute(1,0).view(-1,1)
        filter_scores = filter_scores.view(-1,1)
        objectpairs_scores = detection_scores * filter_scores

        ignore_idx = (torch.ones((num_boxes,num_boxes))-torch.eye(num_boxes)).view(1,-1).squeeze(0)
        remain_idx = ignore_idx ==1
        objectpairs_idx = objectpairs_idx[remain_idx]
        objectpairs_scores = objectpairs_scores[remain_idx]
##
        idx = torch.argsort(objectpairs_scores,dim=0,descending=True)
        idx = idx.view(1,-1).squeeze(0)
        objectpairs_idx = objectpairs_idx[idx]
        objectpairs_scores = objectpairs_scores[idx].view(-1)
##
        if len(objectpairs_scores) !=0:

            subject_boundingboxes = bounding_box[objectpairs_idx[:,0],:]
            object_boundingboxes = bounding_box[objectpairs_idx[:,1],:]
            subject_category = labels[objectpairs_idx[:,0]]
            object_category = labels[objectpairs_idx[:,1]]
            subject_scores = object_scores[objectpairs_idx[:,0]]
            object_scores = object_scores[objectpairs_idx[:,1]]

            xs = torch.min(subject_boundingboxes[:,0],object_boundingboxes[:,0]).view(-1,1)
            ys = torch.min(subject_boundingboxes[:,1],object_boundingboxes[:,1]).view(-1,1)
            xm = torch.max(subject_boundingboxes[:,2],object_boundingboxes[:,2]).view(-1,1)
            ym = torch.max(subject_boundingboxes[:,3],object_boundingboxes[:,3]).view(-1,1)
            boxes = torch.cat((xs,ys,xm,ym),1)
##
            keep = nms(subject_boundingboxes,object_boundingboxes,objectpairs_scores,subject_category,object_category,thresh=0.25)

            boxes = boxes[keep]
            subject_boundingboxes = subject_boundingboxes[keep]
            object_boundingboxes = object_boundingboxes[keep]
            subject_category = subject_category[keep]
            object_category = object_category[keep]
            subject_scores = subject_scores[keep]
            object_scores = object_scores[keep]
            objectpairs_scores = objectpairs_scores[keep]
##

            objectpairs = BoxList(boxes, img_size, mode="xyxy")
            objectpairs.add_field("subject_boundingboxes", subject_boundingboxes)
            objectpairs.add_field("object_boundingboxes", object_boundingboxes)
            objectpairs.add_field("subject_category", subject_category)
            objectpairs.add_field("object_category", object_category)
            objectpairs.add_field("subject_scores", subject_scores)
            objectpairs.add_field("object_scores", object_scores)
            objectpairs.add_field("objectpairs_scores", objectpairs_scores)

            objectpairs_list.append(objectpairs)

            return objectpairs_list

    objectpairs = BoxList(torch.tensor([],device=bounding_box.device).view(-1,4), img_size, mode="xyxy")
    objectpairs.add_field("subject_boundingboxes", torch.tensor([],device=bounding_box.device).view(-1,4))
    objectpairs.add_field("object_boundingboxes", torch.tensor([],device=bounding_box.device).view(-1,4))
    objectpairs.add_field("subject_category", labels.new_empty((0)))
    objectpairs.add_field("object_category", labels.new_empty((0)))
    objectpairs.add_field("subject_scores", labels.new_empty((0)))
    objectpairs.add_field("object_scores", labels.new_empty((0)))
    objectpairs.add_field("objectpairs_scores", labels.new_empty((0)))
    objectpairs_list.append(objectpairs)

    return objectpairs_list