import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.utils import cat
from .objectpair_func import gen_objectpair_targets,gen_objectpair_proposals,combineboxlist,gen_objectpairs
from .new_layers import multi_modal_fusion
from .gnn import GGNN

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
        return x


class predicate_model(nn.Module):

    def __init__(self,cfg,add_targets=True,init_weights=True):
        super(predicate_model, self).__init__()

        self.feature_extractor = make_roi_box_feature_extractor(cfg)

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        feature_dim = cfg.CONFIG.feature_dim

        self.multi_modal_fusion = multi_modal_fusion(cfg)
        self.location_graph = GGNN(cfg,10,1,n_edge_types=1)
        self.criterion = nn.CrossEntropyLoss()

        if init_weights:
            self._initialize_weights()

    def forward(self, features,proposals,targets,devices,**kargs):
        if self.training:
            #targets
            objectpair_targets = gen_objectpair_targets(targets)
            #proposals
            objectpair_proposals = gen_objectpair_proposals(proposals,targets)
            objectpairs,predicate_labels = combineboxlist(objectpair_targets,objectpair_proposals)
            x = self.feature_extractor(features, objectpairs)

            output1 = self.multi_modal_fusion(x,objectpairs,devices)
            output2 = self.location_graph(output1,devices)
            output = output1+output2

            predicate_labels = cat(predicate_labels, dim=0)
            predicate_labels  = predicate_labels.squeeze().to(devices)

            prm_loss = self.criterion(output,predicate_labels)

            return dict(prm_loss=prm_loss)
        else:
            if kargs['eval_criteria'] == "predicate_detection":
                objectpair_targets = gen_objectpair_targets(targets)
                boxes_per_image = [len(box) for box in objectpair_targets]
                if 0 in boxes_per_image:
                    objectpair_targets.pop(boxes_per_image.index(0))
                x = self.feature_extractor(features, objectpair_targets)

                output1 = self.multi_modal_fusion(x,objectpair_targets,devices)
                output2 = self.location_graph(output1,devices)
                output = output1+output2

                output = F.softmax(output,dim=1)
                output = output.split(boxes_per_image, dim=0)
                return output

            else:
                objectpairs = []
                filter_scores = kargs['filter_scores']
                for i,(proposals_per_image,filter_scores_per_image) in enumerate (zip(proposals,filter_scores)):
                    objectpairs_per_image = gen_objectpairs(proposals_per_image,filter_scores_per_image)

                    if len(objectpairs_per_image[0]) !=0:

                        x = self.feature_extractor(features, objectpairs_per_image)
                        output1 = self.multi_modal_fusion(x,objectpairs_per_image,devices)
                        output2 = self.location_graph(output1,devices)
                        output = output1+output2

                        output = F.softmax(output,dim=1)
                    else:
                        output =torch.tensor([])
                    objectpairs_per_image[0].add_field("predicate_scores",output)
                    objectpairs.append(objectpairs_per_image[0])

                return objectpairs                  


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)



