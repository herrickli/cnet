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
    # [xmin, ymin, xmax, ymax]
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


class RPN(nn.Module):
    def __init__(self, anchor_scale=[8, 16, 32], anchor_ratio=[0.5, 1., 2.]):
        # gt_bboxes_info
        super(RPN, self).__init__()
        self.anchor_scale = anchor_scale
        self.anchor_ratio = anchor_ratio
        self.num_anchor = len(self.anchor_scale) * len(self.anchor_ratio) # 9
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1) # the size or feature_map didn't change
        self.score_layer = nn.Conv2d(in_channels=512, out_channels=self.num_anchor*2, kernel_size=1, stride=1, padding=0)
        self.loc_layer = nn.Conv2d(in_channels=512, out_channels=self.num_anchor*4, kernel_size=1, stride=1, padding=0)

        self.min_size = 16

    def forward(self, feature_map, feature_stride):

        batch_size, channels, height, width = feature_map.shape

        anchors = generate_anchor_in_origin_image(feature_height=height, feature_width=width, feature_stride=feature_stride)

        mid_layer = F.relu(self.conv(feature_map))
        rpn_score = self.score_layer(mid_layer) # (1, 2*9, h, w)

        rpn_loc = self.loc_layer(mid_layer)  # (1, 4*9 h, w)
        rpn_loc = rpn_loc.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_score = F.softmax(rpn_score.view(batch_size, height, width, self.num_anchor, 2), dim=4)

        rpn_fg_score = rpn_softmax_score[:, :, :, :, 1].contiguous()
        rpn_fg_score = rpn_fg_score.view(batch_size, -1)
        rpn_fg_score = rpn_fg_score[0].cpu().detach().numpy()

        rpn_score = rpn_score.view(batch_size, -1, 2)

        rois = loc2bbox(anchors, rpn_loc[0].cpu().detach().numpy())
        rois = clip_bbox(rois, height*feature_stride, width*feature_stride)

        # abandon too small box
        ws = rois[:, 2] - rois[:, 0]
        hs = rois[:, 3] - rois[:, 1]
        keep = np.where((ws >= self.min_size) & (hs >= self.min_size))[0]
        rois = rois[keep, :]
        rpn_fg_score = rpn_fg_score[keep]

        #order the rois
        order = rpn_fg_score.ravel().argsort()[::-1]
        rois = rois[order, :]
        rpn_fg_score = rpn_fg_score[order]

        # nms
        keep = nms(rois, rpn_fg_score.reshape(-1), thresh=0.5)
        rois = rois[keep]

        return rois, anchors, rpn_loc, rpn_score



if __name__ == '__main__':
    from pprint import pprint

    gt_boxes = torch.Tensor([[0.,  0., 150., 150.]])
    label = torch.Tensor([1])
    input = torch.randn(1, 2048, 40, 40)
    model = RPN()
    rois = model(input, 16)
