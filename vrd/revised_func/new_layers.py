import torch
import torch.nn as nn
import math
import scipy.io as scio
import os
from maskrcnn_benchmark.modeling.utils import cat

class multi_modal_fusion(nn.Module):

    def __init__(self,cfg):
        super(multi_modal_fusion, self).__init__()
        feature_dim = cfg.CONFIG.feature_dim
        language_dim = cfg.CONFIG.language_dim
        location_dim = cfg.CONFIG.location_dim
        num_predicates = cfg.CONFIG.num_predicates
        self.language= nn.Sequential(
            nn.Linear(language_dim, 100),
            nn.ReLU(),
            nn.Linear(100, feature_dim),
        )
        self.location = nn.Sequential(
            nn.Linear(location_dim, 100),
            nn.ReLU(),
            nn.Linear(100, feature_dim),
        )
        self.visual = nn.Sequential(
            nn.Conv2d(256, 512,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,feature_dim,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(7)
        self.classifier =  nn.Linear(feature_dim,num_predicates)
        self.word2vec = scio.loadmat(os.getcwd()+cfg.CONFIG.word2vec)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()   

    def forward(self,x_visual,objectpairs,devices):

        #visual encoding
        batch_size = x_visual.size(0)
        x_visual = self.visual(x_visual)
        x_visual = self.pool(x_visual)
        x_visual = x_visual.view(batch_size,-1)

        input_language,input_location = compute_multi_modal(objectpairs,self.word2vec,devices)
        #language encoding
        context_lan = self.language(input_language)
        numerator_lan = nn.Softplus()(context_lan) + torch.FloatTensor([0.1]).cuda().expand(context_lan.size())
        dominator_lan = torch.sum(numerator_lan, dim=1)
        lan_encoding = numerator_lan / dominator_lan.unsqueeze(-1)

        #location encoding
        context_loc = self.location(input_location)
        numerator_loc = nn.Softplus()(context_loc) + torch.FloatTensor([0.1]).cuda().expand(context_loc.size())
        dominator_loc = torch.sum(numerator_loc, dim=1)
        loc_encoding = numerator_loc / dominator_loc.unsqueeze(-1)

        x = x_visual*lan_encoding*loc_encoding
        x = self.classifier(x)

        return x

def compute_multi_modal(targets,word2vec,devices):
    word2vec = torch.tensor(word2vec['a']).to(devices)
    category_info =[]
    spatial_info =[]
    eps = torch.tensor(10e-16).to(devices)
    for i,target in enumerate(targets):

        subject_category = (target.get_field('subject_category')-1)
        object_category = (target.get_field('object_category')-1)
        sub = word2vec[subject_category]
        ob = word2vec[object_category]
        w2v = torch.cat((sub,ob),1)
        category_info.append(w2v)

        subject_boundingboxes = target.get_field('subject_boundingboxes')
        object_boundingboxes = target.get_field('object_boundingboxes')
        W,H = target.size
        W = float(W)
        H = float(H)
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
        spatial=torch.cat((x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13),dim=1)
        spatial_info.append(spatial)

    category_info = cat(category_info, dim=0).float()
    spatial_info = cat(spatial_info, dim=0)
    spatial_info = torch.div(spatial_info,torch.norm(spatial_info,dim=1).unsqueeze(-1)).float()

    return category_info,spatial_info