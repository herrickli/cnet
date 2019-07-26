import torch
import torch.optim as optim

from calc_loss import _calc_loc_loss
from dataset import custom_dataset
from resnet import Resnet101
from torch.utils.data import DataLoader
import torch.nn.functional as F

from rpn import RPN
from rpn_target_creator import RPNTargetCreator
from tools import at

def train():
    resnet = Resnet101()
    rpn = RPN()
    rpn_targegt_creator = RPNTargetCreator()
    dataset = custom_dataset()
    data_loader = DataLoader(dataset, batch_size=1 , shuffle=False)
    for img_batch, bndboxes_batch, labels_batch in data_loader:
        # only support batch_size=1 for now
        img, bndboxes, labels = img_batch, bndboxes_batch[0], labels_batch[0]
        batch_size, channels, height, width = img.shape
        out = resnet(img.float())
        rois, anchors, rpn_loc, rpn_score = rpn(out, feature_stride=32)
        gt_rpn_loc, gt_rpn_label = rpn_targegt_creator(anchors, bndboxes.numpy(), (height, width))
        gt_rpn_label = at.toTensor(gt_rpn_label).long()
        gt_rpn_loc = at.toTensor(gt_rpn_loc).float()
        rpn_loc_loss = _calc_loc_loss(rpn_loc[0], gt_rpn_loc, gt_rpn_label, sigma=3.)

        optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)
        optimizer.zero_grad()

        rpn_cls_loss = F.cross_entropy(rpn_score[0], gt_rpn_label, ignore_index=-1)
        rpn_loss = rpn_loc_loss + rpn_cls_loss
        print('current loss:', rpn_loss.item())
        rpn_loss.backward()

        optimizer.step()
        rpn_loss = 0




if __name__ == '__main__':
    train()