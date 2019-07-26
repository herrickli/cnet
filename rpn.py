from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from nms import nms
from tools import bbox_iou, loc2bbox, clip_bbox


def generate_anchor_in_cell(anchor_scale=[8, 16, 32], anchor_ratio=[0.5, 1., 2.], feature_stride=16):
    base_size = feature_stride
    anchor = list()
    for scale in anchor_scale:
        for ratio in anchor_ratio:
            w = (base_size * scale) * np.sqrt(ratio)
            h = w / ratio
            xmin, ymin = base_size / 2 - w / 2,  base_size / 2 - h / 2
            xmax, ymax = xmin + w, ymin + h
            anchor.append([xmin, ymin, xmax, ymax])
    return np.array(anchor)

def generate_anchor_in_origin_image(feature_height, feature_width, base_anchor=None, feature_stride=16):
    base_anchor = generate_anchor_in_cell()
    grid_x = np.arange(0, feature_stride * feature_width, feature_stride)
    grid_y = np.arange(0, feature_stride * feature_height, feature_stride)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid = np.stack((grid_x.flatten(), grid_y.flatten(), grid_x.flatten(),  grid_y.flatten()), axis=1)
    A = base_anchor.shape[0]
    K = grid.shape[0]
    anchor = base_anchor.reshape((1, A, 4)) + grid.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape(K*A, 4)
    return anchor




def generate_label(rois, gt_boxes, gt_labels):
    n_sample = 128
    pos_ratio = 0.25
    pos_iou_thresh = 0.5
    neg_iou_thresh_hi = 0.5
    neg_iou_thresh_lo = 0.1
    pos_roi_per_image = np.round(n_sample * pos_ratio)

    iou = bbox_iou(rois, gt_boxes)
    max_iou, gt_assignment = torch.max(iou, dim=1)
    gt_roi_label = gt_labels[gt_assignment] + 1
    pos_index = np.where(np.array(max_iou) > pos_iou_thresh)[0]
    pos_roi_this_image = int(min(pos_roi_per_image, pos_index.size))
    if pos_index.size > 0:
        pos_index = np.random.choice(pos_index, size=pos_roi_this_image, replace=False)
    neg_index = np.where(np.array(max_iou < neg_iou_thresh_hi) & np.array(max_iou >= neg_iou_thresh_lo))[0]
    neg_roi_this_image = int(min(n_sample - pos_roi_this_image, neg_index.size))
    if neg_index.size > 0:
        neg_index = np.random.choice(neg_index, size=neg_roi_this_image, replace=False)
    keep_index = np.append(pos_index, neg_index)
    gt_roi_label = gt_roi_label[keep_index]
    gt_roi_label[pos_roi_this_image:] = 0
    sample_roi = rois[keep_index]



class RPN(nn.Module):
    def __init__(self, anchor_scale=[8, 16, 32], anchor_ratio=[0.5, 1., 2.]):
        # gt_bboxes_info
        super(RPN, self).__init__()
        self.anchor_scale = anchor_scale
        self.anchor_ratio = anchor_ratio
        self.num_anchor = len(self.anchor_scale) * len(self.anchor_ratio) # 9
        self.conv = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1) # the size or feature_map didn't change
        self.score_layer = nn.Conv2d(in_channels=256, out_channels=self.num_anchor*2, kernel_size=1, stride=1, padding=0)
        self.loc_layer = nn.Conv2d(in_channels=256, out_channels=self.num_anchor*4, kernel_size=1, stride=1, padding=0)

    def forward(self, feature_map, feature_stride):

        batch_size, channels, height, width = feature_map.shape

        anchors = generate_anchor_in_origin_image(feature_height=height, feature_width=width, feature_stride=feature_stride)

        mid_layer = F.relu(self.conv(feature_map))
        rpn_score = self.score_layer(mid_layer) # (1, 2*9, h, w)
        rpn_loc = self.loc_layer(mid_layer)  # (1, 4*9 h, w)

        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous()
        rpn_score = rpn_score.view(batch_size, height, width, self.num_anchor, 2)
        rpn_score = F.softmax(rpn_score, dim=4)
        rpn_loc = rpn_loc.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        rpn_fg_score = rpn_score[:, :, :, :, 1].contiguous().view(batch_size, -1)
        rpn_score = rpn_score.view(batch_size, -1, 2)

        rois = loc2bbox(anchors, rpn_loc)
        rois = clip_bbox(rois, height*feature_stride, width*feature_stride)

        # nms
        rois = nms(rois, rpn_fg_score.view(-1), thresh=0.5)

        """
        # generate label
        # generate_label(rois, gt_boxes, gt_labels)

        rpn_label = torch.full(size = (rois.shape[0], self.gt_bboxes.shape[0]), fill_value=-1)
        iou = bbox_iou(rois, self.gt_bboxes)
        rpn_label = torch.where(iou>0.7, torch.full_like(rpn_label, 1), rpn_label)
        rpn_label = torch.where(iou<0.3, torch.full_like(rpn_label, 0), rpn_label)
        ## TODO 把gt_box对应的最大iou的roi置为1, 以避免某个gt_box没有分配到iou
        #mask = torch.stack((torch.max(iou, dim=0)[1], torch.arange(0,self.gt_bboxes.shape[0])), dim=1)

        # 找到每个roi对应的gt_box的最大iou的值
        # 找到每个roi对应的gt_box的最大iou的下标
        max_iou, gt_assignment = torch.max(iou, dim=1)
        label = labels[gt_assignment]
        """

        return rois, anchors, rpn_loc, rpn_score



if __name__ == '__main__':
    from pprint import pprint

    gt_boxes = torch.Tensor([[0.,  0., 150., 150.]])
    label = torch.Tensor([1])
    input = torch.randn(1, 2048, 40, 40)
    model = RPN()
    rois = model(input, 16)
