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

    # results = VRDResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            with open(file_path, "w") as f:
                json.dump(vrd_results, f)

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
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        ids = prediction.get_field("ids").tolist()

        mapped_labels = [i for i in labels]
        a={}
        a.update(filename=original_id)
        a.update(height=image_height)
        a.update(width=image_width)
        a.update(objects_num=len(prediction))
        objects =[
                {
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
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
    detections['detection_labels'] = []
    detections['detection_bboxes'] = []
    detections['detection_confs'] = []
    detections['detection_ids'] = []
    detection_res = vrd_results['bbox']
    for i in range(len(detection_res)):
        labels = []
        bboxes =[]
        confs = []
        ids = []
        for ii in range(len(detection_res[i]['objects'])):
            x= detection_res[i]['objects'][ii]['bbox'][0]
            y= detection_res[i]['objects'][ii]['bbox'][1]
            w= max(detection_res[i]['objects'][ii]['bbox'][2]-1,0)
            h= max(detection_res[i]['objects'][ii]['bbox'][3]-1,0)
            labels.append([detection_res[i]['objects'][ii]['category_id']])
            confs.append([detection_res[i]['objects'][ii]['score']])
            bboxes.append([int(x),int(y),int(x+w),int(y+h)])
            ids.append([detection_res[i]['objects'][ii]['ids']])

        detections['detection_labels'].append(labels)
        detections['detection_confs'].append(confs)
        detections['detection_bboxes'].append(bboxes)
        detections['detection_ids'].append(ids)
    return detections