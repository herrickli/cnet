import torch
import numpy as np

from config import cfg
from tools import bbox_iou, bbox2loc


class RCNNTargetCreator:
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5,
                 neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, rois, gt_boxes, label):
        'label is range from [0, num_classes], 0 is background'
        iou = bbox_iou(rois, gt_boxes)
        gt_assignment = iou.argmax(axis=1)# 与每个roi的iou最大的gt_box的索引
        max_iou = iou.max(axis=1) # 与每个roi的iou最大的iou值
        gt_roi_label = label[gt_assignment]

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_this_image = int(min(self.n_sample * self.pos_ratio, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_this_image, replace=False)

        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_this_image = int(min(neg_index.size, self.n_sample - pos_roi_this_image))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_this_image:] = 0 # background
        sample_roi = rois[keep_index]

        qq = sample_roi
        pp = gt_boxes[gt_assignment[keep_index]]

        gt_roi_loc = bbox2loc(sample_roi, gt_boxes[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array((0., 0., 0., 0.), np.float32)) / np.array((0.1, 0.1, 0.2, 0.2), np.float32))
        #print('sample roi:', len(sample_roi), sample_roi)
        #print('sample label', gt_roi_label)
        return sample_roi, gt_roi_label, gt_roi_loc


if __name__ == '__main__':
    ctc = RCNNTargetCreator()
    ctc(rois=None, gt_boxes=None)
