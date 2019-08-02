import torch
from torch import nn
import numpy as np

from tools import bbox_iou, bbox2loc


def _unmap(data, length, index, fill):
    if len(data.shape) == 1:
        ret = np.empty(shape=(length,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty(shape=(length,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


class RPNTargetCreator:
    def __init__(self,
                 n_sample=256,
                 pos_ratio=0.5,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

    def __call__(self, anchor, gt_boxes, img_size):
        img_h, img_w = img_size
        ori_num_anchor = len(anchor)
        inside_index = np.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= img_w) &
            (anchor[:, 3] <= img_h)
        )[0]

        anchor = anchor[inside_index]

        argmax_iou, label = self.create_rpn_label(anchor, inside_index, gt_boxes)
        loc = bbox2loc(anchor, gt_boxes[argmax_iou])

        # because we only compute loc and label of inside_index anchor, so we neet to
        # ummap the inside anchor to origin anchor set
        label = _unmap(label, ori_num_anchor, inside_index, fill=-1)
        loc = _unmap(loc, ori_num_anchor, inside_index, fill=0)

        return loc, label

    def create_rpn_label(self, anchor, inside_index, gt_boxes):
        # -1 ignore， 0：negative 1：posotive
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)
        iou = bbox_iou(anchor, gt_boxes)

        # 取得每个anchor与之对应的对大的gt_box的iou的索引及值
        argmax_iou = iou.argmax(axis=1)
        max_iou = iou[np.arange(len(inside_index)), argmax_iou]

        gt_argmax_iou = iou.argmax(axis=0)
        gt_max_iou = iou[gt_argmax_iou, np.arange(gt_boxes.shape[0])]
        gt_argmax_iou = np.where(iou == gt_max_iou)[0]

        label[max_iou >= self.pos_iou_thresh] = 1
        label[gt_argmax_iou] = 1
        label[max_iou < self.neg_iou_thresh] = 0

        n_pos = int(self.n_sample * self.pos_ratio)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_iou, label
