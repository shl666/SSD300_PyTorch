import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
import os
import sys
import tarfile
import collections

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image

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


class VOCDetection(tv.datasets.VOCDetection):
    def __init__(self, root, year='2012', image_set='train', download=False, image_size=voc['min_dim']):
        super(VOCDetection, self).__init__(root, year, image_set, download)
        self.image_size = image_size
        self.class_to_ind = dict(zip(VOC2012_CLASSES, range(len(VOC2012_CLASSES))))
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        width, height = img.size
        # since there are some data files have been damaged, 
        # we give lead it to a another backup datset
        try:
            target = self.parse_voc_xml(
                ET.parse(self.annotations[index]).getroot())
        except:
            annotation_path = '/datasets/ee285f-public/PascalVOC2012/Annotations/'
            annotation_path += self.annotations[index].split('/')[-1]
            target = self.parse_voc_xml(
                ET.parse(annotation_path).getroot())
        img_transform = tv.transforms.Compose([
            tv.transforms.Resize((self.image_size,self.image_size)),
            tv.transforms.ToTensor(),
            ])
        img = img_transform(img)
        label = []
        if type(target['annotation']['object']) == list:
            for obj in target['annotation']['object']:
                name = obj['name']
                bbox = obj['bndbox']

                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                for i, pt in enumerate(pts):
                    cur_pt = int(bbox[pt]) - 1
                    # scale height or width
                    cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                    bndbox.append(cur_pt)
                label_idx = self.class_to_ind[name]
                bndbox.append(label_idx)
                label += [bndbox]
        elif type(target['annotation']['object']) == dict:
            obj = target['annotation']['object']
            name = obj['name']
            bbox = obj['bndbox']

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox[pt]) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            label += [bndbox]
        return img, label, height, width
    
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    target = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        target.append(torch.FloatTensor(sample[1]))
    imgs = torch.stack(imgs, 0)
    return imgs, target