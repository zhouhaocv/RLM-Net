# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from torch.utils.data import Dataset
import json
import os

class VGDataset(Dataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        self.ann_file = json.load(open(ann_file, 'r'))
        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ann_file = [objects for objects in self.ann_file if objects['objects_num'] > 0]
        self.root = root
        self.transforms = transforms

    def __getitem__(self, idx):

        anno = self.ann_file[idx]
        path = anno['filename']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        anno_obj = anno['objects']
        # filter crowd annotations
        # TODO might be better to add an extra field
        # anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno_obj]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes

        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes = [obj["category_id"] for obj in anno_obj]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        ids = [i for i in range(len(classes))]
        ids = torch.tensor(ids)
        target.add_field("ids", ids)

        target = target.clip_to_image(remove_empty=False)
        objects_pairs = [obj for obj in anno['objects_pairs']]
        objects_pairs = torch.tensor(objects_pairs)
        target.add_field("objects_pairs", objects_pairs)

        predicate_label = [[obj] for obj in anno['predicate_label']]
        predicate_label = torch.tensor(predicate_label)
        target.add_field('predicate_label', predicate_label)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target, idx

    def __len__(self):
        return len(self.ann_file)

    def get_img_info(self, index):
        img_data = self.ann_file[index]
        return img_data
