import torch
import torch.nn as nn

#from roi_pooling.functions.roi_pooling import roi_pooling_2d
from resnet import resnet101
from tools import at

from roi_pooling import RoIPooling2D


class RCNN(nn.Module):
    def __init__(self,
                 n_class = 21,#include background
                 roi_pooling_size=7):
        super(RCNN, self).__init__()
        self.n_class = n_class
        self.feature_stride = 16
        self.roi_pooling_szie = roi_pooling_size

        self.classifier = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.rcnn_top = nn.Sequential(resnet101().layer4)

        self.cls_loc = nn.Linear(2048, self.n_class * 4)
        self.cls_score = nn.Linear(2048, self.n_class)
        #self.roi_pooling = nn.AdaptiveAvgPool2d(output_size=self.roi_pooling_szie)
        #self.roi_pooling = roi_pooling_2d
        self.roi_pooling = RoIPooling2D(self.roi_pooling_szie, self.roi_pooling_szie, 1./self.feature_stride)

    def forward(self, rois, feature_map):
        #batch_size, channels, feature_height, feature_width = feature_map.shape

        rois = at.toTensor(rois).float()
        rois = torch.cat((torch.zeros(rois.shape[0], 1), rois), dim=1)
        rois = rois.cuda()
        feature_map = feature_map.cuda()
        pooled_feature = self.roi_pooling(feature_map, rois)
        pooled_feature = pooled_feature.cuda()
        h = self.rcnn_top(pooled_feature).mean(3).mean(2)
        #pooled_feature = self.roi_pooling(feature_map, rois) #shape = torch.Size([124, 2048, 7, 7])
        #pooled_feature = pooled_feature.reshape(pooled_feature.shape[0], -1)
        #h = self.classifier(pooled_feature)
        roi_cls_loc = self.cls_loc(h)
        roi_score = self.cls_score(h)
        return roi_cls_loc, roi_score


if __name__ == '__main__':
    import torch
    import numpy as np

    # Data parameters and fixed-random ROIs
    batch_size = 1
    n_channels = 4
    input_size = (12, 8)
    output_size = (7, 7)
    spatial_scale = 0.6
    rois = torch.FloatTensor([
        [4, 1, 1, 6, 6],
        [0, 6, 2, 7, 11],
        [0, 3, 1, 5, 10],
        [0, 3, 3, 3, 3]
    ])

    # Generate random input tensor
    x_np = np.arange(batch_size * n_channels *
                     input_size[0] * input_size[1],
                     dtype=np.float32)
    x_np = x_np.reshape((batch_size, n_channels, *input_size))
    np.random.shuffle(x_np)

    # torchify and gpu transfer
    x = torch.from_numpy(2 * x_np / x_np.size - 1)
    x = x.cuda()
    rois = rois.cuda()

    # Actual ROIpoling operation
    y = roi_pooling_2d(x, rois, output_size,
                       spatial_scale=spatial_scale)
    print(y.shape)