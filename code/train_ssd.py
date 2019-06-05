import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import nntools as nt

from dataset import *
from ssd import *
from matching import *
from visual import *

##############################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOC2012_CLASSES = (  # always index 0 
#     'background',
    'aeroplane', 'bicycle', 'bird', 'boat','bottle', 
    'bus', 'car', 'cat', 'chair','cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')

VOC2012_ROOT = "../../dataset"

voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
#################################################################

train_dataset = VOCDetection(VOC2012_ROOT, year='2012', image_set='train', download=False)
val_dataset = VOCDetection(VOC2012_ROOT, year='2012', image_set='val', download=False)

class SSD300StatsManager(nt.StatsManager):
    def __init__(self):
        super(SSD300StatsManager, self).__init__()
    def summarize(self):
        loss = super(SSD300StatsManager, self).summarize()
        return {'loss': loss}
    
########## MultiBoxLoss Hyper Parameters #####################
overlap_thresh = 0.5
prior_for_matching=True
bkg_label=0
neg_mining=True
neg_pos=3
neg_overlap=0.5 
encode_target=False
alpha = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
##############################################################

train_loader = td.DataLoader(dataset = train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=detection_collate, pin_memory=True)
val_loader = td.DataLoader(dataset = val_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=detection_collate, pin_memory=True)

lr = 1e-3
SSD300_model = build_ssd300(voc['num_classes'], overlap_thresh, prior_for_matching, bkg_label, 
                            neg_mining, neg_pos, neg_overlap, encode_target, alpha, device)
SSD300_model = SSD300_model.to(device)
adam = torch.optim.Adam(SSD300_model.parameters(), lr=lr)
stats_manager = SSD300StatsManager()
exp = nt.Experiment(SSD300_model, train_loader, val_loader, adam, stats_manager,\
                  output_dir="../weight/SSD300_exp", batch_size = batch_size,\
                  perform_validation_during_training=False)
print('Training on {} ...'.format(device))
exp.run(num_epochs=1)
fig, axes = plt.subplots(figsize=(7,6))
plot(exp, fig, axes)