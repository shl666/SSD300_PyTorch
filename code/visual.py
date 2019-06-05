import nntools
import matplotlib.pyplot as plt
import random
from PIL import Image
from matching import *
import pylab

##############################################################

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

def plt_GroundTruth(img, label,idx, figsize=(5,5), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
        """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(label.shape[0]):
        cls_id = int(label[i][4])
        if cls_id >= 0:
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = int(label[i][0] * height)
            ymin = int(label[i][1] * width)
            xmax = int(label[i][2] * height)
            ymax = int(label[i][3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = VOC2012_CLASSES[cls_id]
            plt.gca().text(xmin, ymin - 2,'{:s}'.format(class_name),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()
    plt.savefig("../img/{}_GroundTruth.png".format(idx))

def plt_Predictions(img, label,idx, figsize=(5,5), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
        """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(label.shape[0]):
        cls_id = int(label[i][4])
        if cls_id >= 0:
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = int(label[i][0] * height)
            ymin = int(label[i][1] * width)
            xmax = int(label[i][2] * height)
            ymax = int(label[i][3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = VOC2012_CLASSES[cls_id]
            score = label[i][5]*100
            plt.gca().text(xmin, ymin - 2,'{:s}: {:.2f}%'.format(class_name,score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()  
    plt.savefig("../img/{}_Prediction.png".format(idx)) 
    
def plot(exp, fig, axes):
    with torch.no_grad():
        axes.clear()
        axes.set_title('SSD300 training loss')
        axes.plot([exp.history[k]['loss'] for k in range(len(exp.history))],label="traininng loss")
        axes.set_xlabel('Global Step')
        axes.set_ylabel('Loss')
        axes.legend()
        
        fig.canvas.draw()
    
    
    plt.savefig("../img/Loss.png")