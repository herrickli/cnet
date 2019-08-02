import numpy as np
import torch


def bbox_iou(box_a, box_b):
    """
    (xmin, ymin, xmax, ymax)
    :param box_a: (N, 4)
    :param gt_boxes: (K, 4)
    :return: (N,K) in which (nth, kth) means the iou between nth roi and kth gt_boxes
    """
    # shape = (N,K,2)

    top_left = np.maximum(box_a[:, None, :2], box_b[:, :2])
    bottom_right = np.minimum(box_a[:, None, 2:], box_b[:, 2:])

    area_intersect = np.prod(bottom_right - top_left, axis=2) * (bottom_right > top_left).all(axis=2)
    area_rois = np.prod(box_a[:, 2:] - box_a[:, :2], axis=1)
    ares_gt_boxes = np.prod(box_b[:, 2:] - box_b[:, :2], axis=1)
    iou = area_intersect / (area_rois[:, None] + ares_gt_boxes - area_intersect)

    return iou


def loc2bbox(anchor, loc):
    """refine anchor according to delta
        [xmin, ymin, xmax, ymax]
        Define: loc is d(delta)
                origin is A(anchor)
                prediction is P
        We got:
                Px = Aw * dx + Ax  |
                Py = Ah * dy + Ay  | ==> center coords
                Pw = exp(dw) * Aw   |
                Ph = exp(dh) * Ah   | ==> width and height
    """
    # support batch_size = 1 for now

    if anchor.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    delta = loc

    width = anchor[:, 2] - anchor[:, 0]
    height = anchor[:, 3] - anchor[:, 1]
    centre_x = width / 2 + anchor[:, 0]
    centre_y = height / 2 + anchor[:, 1]

    dx = delta[:, 0::4]
    dy = delta[:, 1::4]
    dw = delta[:, 2::4]
    dh = delta[:, 3::4]

    dst_x = width[:, None] * dx + centre_x[:, None]
    dst_y = height[:, None] * dy + centre_y[:, None]
    dst_h = np.exp(dh) * height[:, None]
    dst_w = np.exp(dw) * width[:, None]

    dst_box = np.zeros(shape=delta.shape, dtype=delta.dtype)
    dst_box[:, 0::4] = dst_x - 0.5 * dst_w
    dst_box[:, 1::4] = dst_y - 0.5 * dst_h
    dst_box[:, 2::4] = dst_x + 0.5 * dst_w
    dst_box[:, 3::4] = dst_y + 0.5 * dst_h

    return dst_box


def bbox2loc(src_box, dst_box):
    """
    compute loc which is the offset from src_box to dst_box
    [xmin, ymin, xmax, ymax]
    Define : S(src_box): origin_box
             G(dst_box): gt_box
             d(loc): delta
    We got : dx = (Gx - Sx)/Sw
             dy = (Gy - Sy)Sh
             dw = log(Gw/Sw)
             dh = log(Gh/Sh)
    """
    width = src_box[:, 2] - src_box[:, 0]
    height = src_box[:, 3] - src_box[:, 1]
    ctr_x = src_box[:, 0] + 0.5 * width
    ctr_y = src_box[:, 1] + 0.5 * height

    # make sure the value is not 0
    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dst_width = dst_box[:, 2] - dst_box[:, 0]
    dst_height = dst_box[:, 3] - dst_box[:, 1]
    dst_ctr_x = dst_box[:, 0] + 0.5 * dst_width
    dst_ctr_y = dst_box[:, 1] + 0.5 * dst_height

    dx = (dst_ctr_x - ctr_x) / width
    dy = (dst_ctr_y - ctr_y) / height
    dw = np.log(dst_width / width)
    dh = np.log(dst_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()

    return loc


def clip_bbox(boxes, height, width):
    boxes[:, 0::4] = boxes[:, 0::4].clip(0, width - 1)
    boxes[:, 1::4] = boxes[:, 1::4].clip(0, height - 1)
    boxes[:, 2::4] = boxes[:, 2::4].clip(0, width - 1)
    boxes[:, 3::4] = boxes[:, 3::4].clip(0, height - 1)
    return boxes

class arrayTool:

    def toTensor(self, data, cuda=False):
        if isinstance(data, torch.Tensor):
            tensor = data.detach().contiguous()
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        if cuda:
            tensor = tensor.cuda()
        return tensor

    def toNumpy(self, data):
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, torch.Tensor):
            return data.detach().numpy()

at = arrayTool()

if __name__ == '__main__':
    anchor = np.array([[1,1, 10, 10], [4,1,8,10], [10, 10, 20,12]])
    gt_box = np.array([[3,3,10,10]])
    iou = bbox_iou(anchor, gt_box)
    print(iou)
