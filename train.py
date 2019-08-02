import torch
from torch import optim
import torch.nn as nn

from calc_loss import _calc_loc_loss
from dataset import custom_dataset
from rcnn import RCNN
from rcnn_target_creator import RCNNTargetCreator
from resnet import resnet101
from torch.utils.data import DataLoader
import torch.nn.functional as F

from rpn import RPN
from rpn_target_creator import RPNTargetCreator
from tools import at
from visdom_plot import draw_loss_cruve


def train():
    cuda = True
    resume = True

    resnet = resnet101(pretrained=True)
    rpn = RPN()
    rpn_targegt_creator = RPNTargetCreator()
    rcnn_target_creator = RCNNTargetCreator()
    rcnn = RCNN()
    if cuda:
        resnet = resnet.cuda()
        rpn = rpn.cuda()
        rcnn = rcnn.cuda()

    dataset = custom_dataset()
    data_loader = DataLoader(dataset, batch_size=1 , shuffle=True)
    data_iter = iter(data_loader)
    num_epoch = 20
    num_iter = len(dataset) // 1

    start_epoch = 1
    start_iter = 0

    optimizer = optim.SGD([
        {'params': rpn.parameters()},
        {'params': resnet.parameters()},
        {'params': rcnn.parameters()}
    ], lr=0.01, momentum=0.9)

    if resume:
        load_path = 'data/cnet.model.state.2.2499.pkl'
        check_point = torch.load(load_path)
        resnet.load_state_dict(check_point['resnet'])
        rpn.load_state_dict(check_point['rpn'])
        rcnn.load_state_dict(check_point['rcnn'])
        optimizer.load_state_dict(check_point['optimizer'])
        start_epoch = check_point['start_epoch']
        start_iter = check_point['start_iter']

    loss_name = ['rpn_cls_loss', 'rpn_loc_loss', 'rcnn_cls_loss', 'rcnn_loc_loss', 'total_loss']
    vis_r_l_loss, vis_r_c_loss, vis_o_l_loss, vis_o_c_loss, vis_to_loss = 0, 0, 0, 0, 0

    for epoch in range(start_epoch, num_epoch):
        for i in range(start_iter, num_iter):
            try:
                img_batch, bndboxes_batch, labels_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                img_batch, bndboxes_batch, labels_batch = next(data_iter)
            # only support batch_size=1 for now
            img, bndboxes, labels = img_batch, bndboxes_batch[0], labels_batch[0]
            batch_size, channels, height, width = img.shape
            # rpn
            if cuda:
                out = resnet(img.float().cuda())
            else:
                out = resnet(img.float())

            rois, anchors, rpn_loc, rpn_score = rpn(out, feature_stride=16)

            gt_rpn_loc, gt_rpn_label = rpn_targegt_creator(anchors, bndboxes.cpu().detach().numpy(), (height, width))
            gt_rpn_label = at.toTensor(gt_rpn_label).long()
            gt_rpn_loc = at.toTensor(gt_rpn_loc).float()

            if cuda:
                gt_rpn_loc = gt_rpn_loc.cuda().float()
                gt_rpn_label = gt_rpn_label.cuda().long()

            rpn_loc_loss = _calc_loc_loss(rpn_loc[0], gt_rpn_loc, gt_rpn_label, sigma=3.)

            rpn_cls_loss = F.cross_entropy(rpn_score[0], gt_rpn_label, ignore_index=-1)

            rpn_loss = rpn_loc_loss + rpn_cls_loss

            sample_roi, gt_roi_label, gt_roi_loc = rcnn_target_creator(rois, bndboxes.cpu().numpy(), labels)
            roi_cls_loc, roi_score = rcnn(sample_roi, out)

            num_rois = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(num_rois, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, num_rois).long(), at.toTensor(gt_roi_label).long()]
            
            gt_roi_loc = at.toTensor(gt_roi_loc).float()
            gt_roi_label = at.toTensor(gt_roi_label).long()

            if cuda:
                gt_roi_loc = gt_roi_loc.cuda().float()
                gt_roi_label = gt_roi_label.cuda().long()

            roi_loc_loss = _calc_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, sigma=1.)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)
            rcnn_loss = roi_cls_loss + roi_loc_loss

            total_loss = rpn_loss + rcnn_loss
            print('rpn_cls_loss:', rpn_cls_loss.item())
            print('rpn_loc_loss:', rpn_loc_loss.item())
            print('rcnn_cls_loss:', roi_cls_loss.item())
            print('rcnn_loc_loss:', roi_loc_loss.item())
            print(i, ': total_loss:', total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            vis_r_c_loss += rpn_cls_loss.item()
            vis_r_l_loss += rpn_loc_loss.item()
            vis_o_c_loss += roi_cls_loss.item()
            vis_o_l_loss += roi_loc_loss.item()
            vis_to_loss  += total_loss.item()


            # draw loss line
            if (i + 1) % 40 == 0:
                vis_loss_value = [vis_r_c_loss, vis_r_l_loss, vis_o_c_loss, vis_o_l_loss, vis_to_loss]
                draw_loss_cruve(loss_name, 2501 * epoch + i, vis_loss_value)
                vis_r_l_loss, vis_r_c_loss, vis_o_l_loss, vis_o_c_loss = 0, 0, 0, 0

            if (i+1) % 500 == 0:
                model_save_path = 'data/cnet.model.state.{}.{}.pkl'.format(epoch, i)
                print('saving model..', model_save_path)
                model_state = {'resnet': resnet.state_dict(),
                    'rpn': rpn.state_dict(),
                    'rcnn': rcnn.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'start_epoch': epoch,
                    'start_iter': i}
                torch.save(model_state, model_save_path)
        start_iter = 0
        if epoch % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            model_state = {'resnet': resnet.state_dict(),
                           'rpn': rpn.state_dict(),
                           'rcnn': rcnn.state_dict(),
                           'optimizer': optimizer.state_dict(),
                           'start_epoch': epoch,
                           'start_iter': 0}
            model_save_path = 'data/cnet.model.state.{}.0.pkl'.format(epoch)
            print('module saved in {}'.format(model_save_path))
            torch.save(model_state, model_save_path)

if __name__ == '__main__':
    train()