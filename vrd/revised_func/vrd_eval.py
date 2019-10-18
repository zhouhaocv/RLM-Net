import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import json
import scipy.io as scio

def do_vrd_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    if box_only:
        logger.info("box_only function not finsihed")
        return
    logger.info("Preparing results for VRD format")
    vrd_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        vrd_results["bbox"] = prepare_for_vrd_detection(predictions, dataset)
    logger.info("Generating MAT format")
    detections = generate_for_vrd_mat(vrd_results)
    file_path = os.path.join(output_folder, "ObjectDet.mat")
    scio.savemat(file_path, detections)
    logger.info("Finsihed")
    return vrd_results


def prepare_for_vrd_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    vrd_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.ann_file[image_id]['filename']
        # if len(prediction) == 0:
        #     continue

        # TODO replace with get_img_info?
        image_width = dataset.ann_file[image_id]["width"]
        image_height = dataset.ann_file[image_id]["height"]

        subject_boundingboxes = prediction.get_field("subject_boundingboxes")
        object_boundingboxes = prediction.get_field("object_boundingboxes")
        prediction_size = prediction.size

        prediction_sub = BoxList(subject_boundingboxes, prediction_size, mode="xyxy")
        prediction_ob = BoxList(object_boundingboxes, prediction_size, mode="xyxy")

        prediction = prediction.resize((image_width, image_height))
        prediction_sub = prediction_sub.resize((image_width, image_height))
        prediction_ob = prediction_ob.resize((image_width, image_height))
        prediction_sub = prediction_sub.convert("xywh")
        prediction = prediction.convert("xywh")
        prediction_ob = prediction_ob.convert("xywh")

        boxes = prediction.bbox.tolist()
        subject_boundingboxes = prediction_sub.bbox.tolist()
        object_boundingboxes = prediction_ob.bbox.tolist()
        subject_category = prediction.get_field("subject_category").tolist()
        object_category = prediction.get_field("object_category").tolist()
        subject_scores = prediction.get_field("subject_scores").tolist()
        object_scores = prediction.get_field("object_scores").tolist()
        objectpairs_scores = prediction.get_field("objectpairs_scores").tolist()
        predicate_scores = prediction.get_field("predicate_scores").tolist()

        ids = prediction.get_field("ids").tolist()

        a={}
        a.update(filename=original_id)
        a.update(height=image_height)
        a.update(width=image_width)
        a.update(objects_num=len(prediction))
        objects =[
                {
                    "subject_boundingboxes": subject_boundingboxes[k],
                    "object_boundingboxes": object_boundingboxes[k],
                    "subject_category": subject_category[k],
                    "object_category": object_category [k],
                    "subject_scores": subject_scores[k],
                    "object_scores": object_scores[k],
                    "objectpairs_scores": objectpairs_scores[k],
                    "predicate_scores": predicate_scores[k],
                    "ids": ids[k],
                }
                for k, box in enumerate(boxes)
                ]
        a.update(objects=objects)
        vrd_results.append(a)
    return vrd_results

def generate_for_vrd_mat(vrd_results):
    # assert isinstance(dataset, COCODataset)
    detections = {}
    detections['subject_boundingboxes'] = []
    detections['object_boundingboxes'] = []
    detections['subject_category'] = []
    detections['object_category'] = []
    detections['subject_scores'] = []
    detections['object_scores'] = []
    detections['objectpairs_scores'] = []
    detections['predicate_scores'] = []
    detections['ids'] = []
    detection_res = vrd_results['bbox']
    for i in range(len(detection_res)):
        subject_boundingboxes = []
        object_boundingboxes =[]
        subject_category = []
        object_category = []
        subject_scores = []
        object_scores =[]
        objectpairs_scores = []
        predicate_scores = []
        ids = []
        for ii in range(len(detection_res[i]['objects'])):
            x_s= detection_res[i]['objects'][ii]['subject_boundingboxes'][0]
            y_s= detection_res[i]['objects'][ii]['subject_boundingboxes'][1]
            w_s= max(detection_res[i]['objects'][ii]['subject_boundingboxes'][2]-1,0)
            h_s= max(detection_res[i]['objects'][ii]['subject_boundingboxes'][3]-1,0)
            subject_boundingboxes.append([int(x_s),int(y_s),int(x_s+w_s),int(y_s+h_s)])

            x_o= detection_res[i]['objects'][ii]['object_boundingboxes'][0]
            y_o= detection_res[i]['objects'][ii]['object_boundingboxes'][1]
            w_o= max(detection_res[i]['objects'][ii]['object_boundingboxes'][2]-1,0)
            h_o= max(detection_res[i]['objects'][ii]['object_boundingboxes'][3]-1,0)
            object_boundingboxes.append([int(x_o),int(y_o),int(x_o+w_o),int(y_o+h_o)])

            subject_category.append([detection_res[i]['objects'][ii]['subject_category']])
            object_category.append([detection_res[i]['objects'][ii]['object_category']])
            subject_scores.append([detection_res[i]['objects'][ii]['subject_scores']])
            object_scores.append([detection_res[i]['objects'][ii]['object_scores']])
            objectpairs_scores.append([detection_res[i]['objects'][ii]['objectpairs_scores']])
            predicate_scores.append(detection_res[i]['objects'][ii]['predicate_scores'])
            ids.append([detection_res[i]['objects'][ii]['ids']])

        detections['subject_boundingboxes'].append(subject_boundingboxes)
        detections['object_boundingboxes'].append(object_boundingboxes)
        detections['subject_category'].append(subject_category)
        detections['object_category'].append(object_category)
        detections['subject_scores'].append(subject_scores)
        detections['object_scores'].append(object_scores)
        detections['objectpairs_scores'].append(objectpairs_scores)
        detections['predicate_scores'].append(predicate_scores)
        detections['ids'].append(ids)
    return detections