import math
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.poolers import Pooler
from .loss import make_rating_model_loss_evaluator
from .objectpair_func import compute_spatial_objectpairs
from .objectpair_func import compute_labels,compute_proposals_labels,BalancedPN
from maskrcnn_benchmark.modeling.utils import cat

class make_roi_box_feature_extractor(nn.Module):
    def __init__(self, cfg):
        super(make_roi_box_feature_extractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        if x.size(0) !=0:
            x = x.view(x.size(0), -1)
        else:
            x= torch.tensor([],device=x.device).reshape(0,12544)
        return x


class rating_model(nn.Module):

    def __init__(self,cfg,add_targets=True,init_weights=True):
        super(rating_model, self).__init__()

        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.loss_evaluator = make_rating_model_loss_evaluator(cfg)
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048+14, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )
        self.add_targets = add_targets
        if init_weights:
            self._initialize_weights()

    def forward(self, features,proposals,targets,devices):
        if self.training:
            if self.add_targets:
                x_proposals = self.feature_extractor(features, proposals)
                x_targets = self.feature_extractor(features, targets)
                x= torch.cat((x_proposals,x_targets),0)
                x = self.fc(x)
                split_box = [len(box) for i, box in enumerate (proposals)]
                split_box.extend([len(box) for i, box in enumerate (targets)])

            num_images = len(proposals)
            x = x.split(split_box, dim=0)
            xs =[]
            labels = []
            for i,(targets_per_image,proposals_per_image) in enumerate (zip(targets,proposals)):
                x_per_image_proposals = x[i]
                x_per_image_targets =x[i+num_images]
            
                ##proposals
                if len(x_per_image_proposals) !=0:
                    x_subject_proposals = x_per_image_proposals.unsqueeze(1).expand(x_per_image_proposals.size(0),x_per_image_proposals.size(0),x_per_image_proposals.size(1))
                    x_object_proposals = x_per_image_proposals.unsqueeze(0).expand(x_per_image_proposals.size(0),x_per_image_proposals.size(0),x_per_image_proposals.size(1))
                    x_proposals=torch.cat((x_subject_proposals,x_object_proposals),2)
                    x_proposals=x_proposals.view(-1,2*x_per_image_proposals.size(1))
                    x_proposals = torch.div(x_proposals,torch.norm(x_proposals,dim=1).unsqueeze(-1))

                    spatial_info_proposals = compute_spatial_objectpairs(proposals_per_image)
                    spatial_info_proposals =spatial_info_proposals.to(devices)

                    x_proposals = torch.cat((x_proposals,spatial_info_proposals),1)
                    labels_per_image_proposals,mask_proposals = compute_proposals_labels(proposals_per_image,targets_per_image)
                    remain_idx_proposals = mask_proposals ==1
                    labels_per_image_proposals = labels_per_image_proposals[remain_idx_proposals]
                    x_proposals = x_proposals[remain_idx_proposals]

                else:
                    x_proposals = torch.tensor([])
                    labels_per_image_proposals = torch.tensor([])

                ##targets
                x_subject_targets = x_per_image_targets.unsqueeze(1).expand(x_per_image_targets.size(0),x_per_image_targets.size(0),x_per_image_targets.size(1))
                x_object_targets = x_per_image_targets.unsqueeze(0).expand(x_per_image_targets.size(0),x_per_image_targets.size(0),x_per_image_targets.size(1))
                x_targets=torch.cat((x_subject_targets,x_object_targets),2)
                x_targets=x_targets.view(-1,2*x_per_image_targets.size(1))
                x_targets = torch.div(x_targets,torch.norm(x_targets,dim=1).unsqueeze(-1))
                
                spatial_info_targets = compute_spatial_objectpairs(targets_per_image)
                spatial_info_targets =spatial_info_targets.to(devices)

                x_targets = torch.cat((x_targets,spatial_info_targets),1)
                labels_per_image_targets,mask_targets = compute_labels(targets_per_image)
                remain_idx_targets = mask_targets ==1
                labels_per_image_targets = labels_per_image_targets[remain_idx_targets]
                x_targets = x_targets[remain_idx_targets]


                x1,labels_per_image = BalancedPN(x_proposals,x_targets,labels_per_image_proposals,labels_per_image_targets,devices)

                xs.append(x1)
                labels.append(labels_per_image)


            x = cat(xs, dim=0)
            labels = cat(labels, dim=0)
            labels = labels.to(devices)
            outputs = self.fc2(x)
            outputs = outputs.squeeze()

            rating_model_loss = F.binary_cross_entropy_with_logits(outputs, labels)

            return dict(orm_loss=rating_model_loss)
        else:
            x = self.feature_extractor(features, proposals)
            x = self.fc(x)

            boxes_per_image = [len(box) for box in proposals]
            x = x.split(boxes_per_image, dim=0)
            outputs =[]
            for i,(x_per_image,proposals_per_image) in enumerate (zip(x,proposals)):
                if len(x_per_image) !=0:
                    x_subject = x_per_image.unsqueeze(1).expand(x_per_image.size(0),x_per_image.size(0),x_per_image.size(1))
                    x_object = x_per_image.unsqueeze(0).expand(x_per_image.size(0),x_per_image.size(0),x_per_image.size(1))
                    x=torch.cat((x_subject,x_object),2)
                    x=x.view(-1,2*x_per_image.size(1))
                    x = torch.div(x,torch.norm(x,dim=1).unsqueeze(-1))

                    spatial_info = compute_spatial_objectpairs(proposals_per_image)
                    spatial_info =spatial_info.to(devices)


                    x = torch.cat((x,spatial_info),1)

                    x=self.fc2(x)
                    output_per_image = torch.sigmoid(x)
                    output_per_image = output_per_image.view(x_per_image.size(0),x_per_image.size(0))
                else:
                    output_per_image = torch.tensor([]).reshape(x_per_image.size(0),x_per_image.size(0))
                outputs.append(output_per_image)

            return outputs


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)



