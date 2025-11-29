import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino.dinov2 import vit_small, vit_base, vit_large, vit_giant2
from models.module import *
from models.FMT import FMT_with_pathway_trimnet
import os
import torch.backends.cudnn as cudnn
import sys
from models.dpt_trimnet import DPTHead_decoder
from models.cas_model import FeatExt_trimnet,RefineNet_residual
Align_Corners_Range = False

def masked_mse_loss(pred, target, mask):
    mse = nn.MSELoss()
    masked_pred = pred[mask]
    masked_target = target[mask]
    loss = mse(masked_pred, masked_target)
    return loss

def masked_l1_loss(pred, target, mask):
    masked_pred = pred[mask]
    masked_target = target[mask]
    loss = F.smooth_l1_loss(masked_pred, masked_target,reduction='mean')
    return loss

def masked_bce_loss(pred, target, mask):
    bce = nn.BCELoss()
    masked_pred = pred[mask]
    masked_target = target[mask]
    loss = bce(masked_pred, masked_target)
    return loss

class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with



class DINOv2Trimnet(nn.Module):
    def __init__(self, config):
        super(DINOv2Trimnet, self).__init__()
        self.config = config
        self.feat_net = FeatExt_trimnet(input_channels = 6)
        self.feat_net.depth_head = RefineNet_residual(32)


    def forward(self, img, depth_pm, cost_pm):

        feat_input = torch.concatenate([img, depth_pm, cost_pm,(depth_pm>0).float()], dim=1)

        out = self.feat_net(feat_input)
        # need to add clip

        attention_out = []

        attention_out.append(F.sigmoid(out[0]))
        attention_out.append(F.sigmoid(out[1]))
        attention_out.append(F.sigmoid(out[2]))
        return attention_out




def initialize_unfrozen_weights(module):
    if isinstance(module, nn.Conv2d):
        if module.weight.requires_grad:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None and module.bias.requires_grad:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        if module.weight.requires_grad:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None and module.bias.requires_grad:
            nn.init.constant_(module.bias, 0)
