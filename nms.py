import torch
import numpy as np

def nms(rois, _scores, thresh):
    xmin, ymin, xmax, ymax = rois[:,0], rois[:,1], rois[:,2], rois[:,3]
    scores = _scores
    order = scores.argsort()[::-1]

    areas = (xmax - xmin) * (ymax - ymin)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(xmin[i], xmin[order[1:]])
        yy1 = np.maximum(ymin[i], ymin[order[1:]])
        xx2 = np.minimum(xmax[i], xmax[order[1:]])
        yy2 = np.minimum(ymax[i], ymax[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        left_index = np.where(iou <= thresh)[0]

        order = order[left_index + 1]

    return keep

if __name__ == '__main__':
    dets = [[0, 0, 100, 101], [5, 6, 90, 110], [17, 19, 80, 120], [10, 8, 115, 105]]
    dets = np.array(dets)
    scores = np.array([0.9, 0.1, 1.0, 0.5])
    keep = nms(dets, scores, 0.5)

    print(dets[keep])

