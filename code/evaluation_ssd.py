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

from collections import defaultdict
import itertools
import numpy as np
import six

def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value..
    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.
        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.
        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.
    """

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


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
SSD300_model.eval()
#############################################################
print("Evaluating on validation set")
pred_bboxes = []
pred_labels = []
pred_scores = []
gt_bboxes = []
gt_labels = []
SSD300_model.eval()
for i_batch, sample_batched in enumerate(val_loader):
    print("Mini-Batch NO.:", i_batch)
    x,target = sample_batched
    x = x.to(device)
    LOC, CONF, prior = SSD300_model.forward(x)
    prior = prior.to(device)
    for i in range(LOC.shape[0]):
        predict = torch.zeros(1,6)
        loc, conf= LOC[i], CONF[i]
        softmax = F.softmax(conf.data,1)
        # decode loc to (x_min, y_min, x_max, y_max)
        decode_loc = decode(loc, prior, SSD300_model.MultiBoxLoss.variance).view(1,-1,4)
        for idx in range(1,21):
            thresh = 0.9
            dets = softmax[:,idx]
            mask_score = dets.gt(thresh).t()
            mask_bbox = dets.gt(thresh).expand(4, dets.size(0)).t()
            score = torch.masked_select(dets, mask_score).view(-1)
            bbox = torch.masked_select(decode_loc.data, mask_bbox).view(-1, 4)
            if bbox.shape[0]!=0:
                keep, count = nms(bbox, score, overlap=0.5, top_k=200)
                class_scores = score[keep[:count]].view(-1,1).to('cpu')
                class_boxes = torch.clamp(bbox[keep[:count]],0,1).to('cpu')
                tmp = torch.ones(count,1)*(idx-1)
                tmp = torch.cat((tmp, class_scores), 1)
                tmp = torch.cat((class_boxes, tmp), 1)
                predict = torch.cat((predict, tmp), 0)
        predict = predict[1:]
        pred_bboxes.append(predict[:,:-2].numpy())
        pred_labels.append(predict[:,-2].numpy())
        pred_scores.append(predict[:,-1].numpy())
        gt_bboxes.append(target[i][:,:4].numpy())
        gt_labels.append(target[i][:,-1].numpy())
        
prec,rec = calc_detection_voc_prec_rec(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels)
mAP_matrix = calc_detection_voc_ap(prec,rec)
mAP = mAP_matrix.mean()
print("mAP Matrix:")
print(mAP_matrix)
print("mAP = ",mAP)
