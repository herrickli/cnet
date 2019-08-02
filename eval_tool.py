from collections import defaultdict

import torch
import numpy as np

from tools import bbox_iou


def calc_precision_and_recall(pred_boxes,
                              pred_labels,
                              pred_scores,
                              gt_boxes,
                              gt_labels,
                              iou_thresh=0.5):
    pred_boxes = iter(pred_boxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_boxes = iter(gt_boxes)
    gt_labels = iter(gt_labels)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_box, pred_score, pred_label, gt_box, gt_label in zip(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):

        gt_difficult = np.zeros(gt_box.shape[0], dtype=bool) # false

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_box_l = pred_box[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            # sort
            order = pred_score_l.argsort()[::-1]
            pred_box_l = pred_box_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_box_l = gt_box[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_box_l) == 0:
                continue
            if len(gt_box_l) == 0:
                match[l].extend((0,) * pred_box_l.shape[0])
                continue

            pred_box_l = pred_box_l.copy()
            pred_box_l[:, 2:] += 1
            gt_box_l = gt_box_l.copy()
            gt_box_l[:, 2:] += 1

            iou = bbox_iou(pred_box_l, gt_box_l)
            gt_index = iou.argmax(axis=1)
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_box_l.shape[0], dtype=bool)

            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterable should be same.')

    n_fg_class = max(n_pos.keys()) + 1 # all classes, exclude background
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (tp + fp)
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec

def calc_average_precision(prec, rec):

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        mprec = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
        mrec = np.concatenate(([0], rec[l], [1]))

        # ap的计算方法是对于m个正例有m个recall值r'，当racall>r'时取最大的precision
        mprec = np.maximum.accumulate(mprec[::-1])[::-1]

        # 对mrec向后移动一位后比较，找出变化的位置
        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mprec[i + 1])

    return ap


def calc_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):

    prec, rec = calc_precision_and_recall(pred_boxes,
                                          pred_labels,
                                          pred_scores,
                                          gt_boxes,
                                          gt_labels,
                                          iou_thresh)

    ap = calc_average_precision(prec, rec)

    return {'ap': ap, 'mAP': np.nanmean(ap)}