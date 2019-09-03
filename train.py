import torch
from torch import optim
import torch.nn as nn

from calc_loss import _calc_loc_loss
from config import cfg
from dataset import custom_dataset
from rcnn import RCNN
from rcnn_target_creator import RCNNTargetCreator
from resnet import resnet101
from torch.utils.data import DataLoader
import torch.nn.functional as F

from rpn import RPN
from rpn_target_creator import RPNTargetCreator
from tools import at, normal_init, loc2bbox
from visdom_plot import draw_loss_cruve


def train():
    cuda = True
    resume = True

    resnet = resnet101()
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

    lr = 0.001
    params = []
    for net in [resnet, rpn, rcnn]:
        for key, value in dict(net.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.weight_decay}]

    optimizer = optim.SGD(params, momentum=0.9)

    if resume:
        rpn_check_point = torch.load("/home/licheng/home/licheng/projects/cnet/data/rpn/rpn_epoch_19.params")
        rpn.load_state_dict(rpn_check_point['rpn'])
        resnet.load_state_dict(rpn_check_point['resnet'])

        rcnn_check_point = torch.load("/home/licheng/home/licheng/projects/cnet/data/rcnn/rcnn_epoch_19.params")
        rcnn.load_state_dict(rcnn_check_point['rcnn'])

        """
        rpn_load_path = '/home/licheng/home/licheng/projects/cnet/data/rpn/rpn_epoch_19.params'
        rpn_check_point = torch.load(rpn_load_path)
        #resnet.load_state_dict(rpn_check_point['resnet'])
        rpn.load_state_dict(rpn_check_point['rpn'])

        rcnn_load_path = '/home/licheng/home/licheng/projects/cnet/data/rcnn/rcnn_epoch_19.params'
        rcnn_check_point = torch.load(rcnn_load_path)
        rcnn.load_state_dict(rcnn_check_point['rcnn'])
        """
    loss_name = ['rpn_cls_loss', 'rpn_loc_loss', 'rcnn_cls_loss', 'rcnn_loc_loss', 'total_loss']
    vis_r_l_loss, vis_r_c_loss, vis_o_l_loss, vis_o_c_loss, vis_to_loss, vis_to_loss = 0, 0, 0, 0, 0, 0

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
                draw_loss_cruve(loss_name, 2501 * (epoch-1) + i, vis_loss_value)
                vis_r_l_loss, vis_r_c_loss, vis_o_l_loss, vis_o_c_loss, vis_to_loss = 0, 0, 0, 0, 0

        model_save_path = 'data/cnet.model.state.{}.pkl'.format(epoch)
        print('saving model..', model_save_path)
        model_state = {'resnet': resnet.state_dict(),
            'rpn': rpn.state_dict(),
            'rcnn': rcnn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'start_epoch': epoch}
        torch.save(model_state, model_save_path)
        start_iter = 0
        if epoch % 9 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    model_state = {'resnet': resnet.state_dict(),
                   'rpn': rpn.state_dict(),
                   'rcnn': rcnn.state_dict()}
    model_save_path = 'data/final.pkl'
    torch.save(model_state, model_save_path)

def train_rpn():
    """
    I think the accuracy of rpn is good enough
    :return:
    """

    resnet = resnet101(pretrained=True).cuda()

    rpn = RPN().cuda()


    #load_path = 'data/rpn/rpn_19.2499'
    #check_point = torch.load(load_path)

    #rpn.load_state_dict(check_point['rpn'])
    #resnet.load_state_dict(check_point['resnet'])

    rpn_targegt_creator = RPNTargetCreator()
    dataset = custom_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataiter = iter(dataloader)
    total_epoch = 20

    lr = 0.01
    params = []
    for net in [resnet, rpn]:
        for key, value in dict(net.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.weight_decay}]
    optimizer = optim.SGD(params, momentum=0.9)

    loss_name = ['TRAIN RPN rpn_cls_loss', 'TRAIN RPN rpn_loc_loss']
    vis_rpn_loc_loss, vis_rpn_cls_loss= 0, 0

    for epoch in range(total_epoch):
        for i in range(0, len(dataset)):
            try:
                img_batch, bndboxes_batch, labels_batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                img_batch, bndboxes_batch, labels_batch = next(dataiter)
            # only support batch_size=1 for now
            img, bndboxes, labels = img_batch, bndboxes_batch[0], labels_batch[0]
            img, bndboxes, labels = img.cuda(), bndboxes.cuda(), labels.cuda()
            batch_size, channels, height, width = img.shape
            feature = resnet(img)
            rois, anchors, rpn_loc, rpn_score = rpn(feature, feature_stride=16)
            gt_rpn_loc, gt_rpn_label = rpn_targegt_creator(anchors, bndboxes.cpu().detach().numpy(), (height, width))

            gt_rpn_label = at.toTensor(gt_rpn_label).long().cuda()
            gt_rpn_loc = at.toTensor(gt_rpn_loc).float().cuda()

            rpn_loc_loss = _calc_loc_loss(rpn_loc[0], gt_rpn_loc, gt_rpn_label, sigma=3.)

            rpn_cls_loss = F.cross_entropy(rpn_score[0], gt_rpn_label, ignore_index=-1)

            rpn_loss = rpn_loc_loss + rpn_cls_loss
            print('rpn_loc_loss:', rpn_loc_loss.item(), 'rpn_cls_loss:', rpn_cls_loss.item())
            print('total_loss:', rpn_loss.item())
            optimizer.zero_grad()

            rpn_loss.backward()

            optimizer.step()

            vis_rpn_cls_loss += rpn_cls_loss.item()
            vis_rpn_loc_loss += rpn_loc_loss.item()

            if (i + 1) % 40 == 0:
                vis_loss_value = [vis_rpn_cls_loss, vis_rpn_loc_loss]
                draw_loss_cruve(loss_name, 2501 * (epoch - 1) + i, vis_loss_value)
                vis_rpn_loc_loss, vis_rpn_cls_loss= 0, 0

        if (epoch + 1) % 10 == 0:
            for parameter_group in optimizer.param_groups:
                parameter_group['lr'] *= 0.1

        svae_path = 'data/rpn/rpn_epoch_{}.params'.format(epoch)
        state_dict = {'rpn': rpn.state_dict(),
                      'resnet': resnet.state_dict()}
        torch.save(state_dict, svae_path)


def train_rcnn():
    """ I think the coord accuracy of rcnn is good enough"""
    resnet = resnet101(pretrained=False)
    rpn = RPN()
    rcnn = RCNN()

    resnet.cuda()
    rpn.cuda()
    rcnn.cuda()
    resume = True
    if resume:
        check_point = torch.load('/home/licheng/home/licheng/projects/cnet/data/rpn/rpn_epoch_19.params')
        rpn.load_state_dict(check_point['rpn'])
        resnet.load_state_dict(check_point['resnet'])
        #rcnn_check_point = torch.load("/home/licheng/home/licheng/projects/cnet/data/rcnn/rcnn_epoch_16.params")
        #rcnn.load_state_dict(rcnn_check_point['rcnn'])

        #rcnn_check_point = torch.load('/home/licheng/home/licheng/projects/cnet/data/rcnn/rcnn_epoch_5.params')
        #rcnn.load_state_dict(rcnn_check_point['rcnn'])

    # fix prarams


    params = []
    lr = 0.01
    for net in [rcnn]:
        for key, value in dict(net.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.weight_decay}]

    rcnn_optimizer = optim.SGD(params, momentum=0.9)

    rcnn_target_creator = RCNNTargetCreator()

    dataset = custom_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataiter = iter(dataloader)
    total_epoch = 20

    loss_name = ['TRAIN RCNN rcnn_cls_loss', 'TRAIN RCNN rcnn_loc_loss']
    vis_o_l_loss, vis_o_c_loss= 0, 0


    for epoch in range(total_epoch):
        for i in range(len(dataset)):
            try:
                img_batch, bndboxes_batch, labels_batch = next(dataiter)
            except StopIteration:
                dataiter = iter(dataloader)
                img_batch, bndboxes_batch, labels_batch = next(dataiter)
            img, bndboxes, labels = img_batch, bndboxes_batch[0], labels_batch[0]
            img, bndboxes, labels = img.cuda(), bndboxes.cuda(), labels.cuda()
            feature = resnet(img)
            rois, _, _, _ = rpn(feature, feature_stride=16)

            #print('gt boxes:', bndboxes)
            sample_roi, gt_roi_label, gt_roi_loc = rcnn_target_creator(rois, bndboxes.cpu().numpy(), labels)
            #print('pred box', len(sample_roi), sample_roi)
            roi_cls_loc, roi_score = rcnn(sample_roi, feature)

            # apple roi_cls_loc to sample_rois has a good result
            mean = torch.Tensor((0., 0., 0., 0.)).repeat(cfg.n_class)[None].cuda()
            std  = torch.Tensor((0.1, 0.1, 0.2, 0.2)).repeat(cfg.n_class)[None].cuda()
            loc = (roi_cls_loc * std + mean)

            look_score1 = at.toNumpy(roi_score.detach().cpu())
            look_score2 = at.toNumpy(F.softmax(roi_score, dim=1).detach().cpu())

            num_rois = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.view(num_rois, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, num_rois).long(), at.toTensor(gt_roi_label).long()]

            keep = gt_roi_label > 0
            keep = at.toNumpy(keep.detach().cpu())

            loc = loc.view(num_rois, -1, 4)
            loc = loc[torch.arange(0, num_rois).long(), at.toTensor(gt_roi_label).long()]

            pred_boxes = loc2bbox(sample_roi, at.toNumpy(loc.detach().cpu()))[keep>0, :]

            gt_roi_loc = at.toTensor(gt_roi_loc).float().cuda()
            gt_roi_label = at.toTensor(gt_roi_label).long().cuda()

            roi_loc_loss = _calc_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, sigma=1.)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)
            rcnn_loss = roi_cls_loss + roi_loc_loss

            print('rcnn_loc_loss:', roi_loc_loss.item(), 'rcnn_cls_losss:', roi_cls_loss.item())
            print('total_loss', rcnn_loss.item())
            rcnn_optimizer.zero_grad()

            rcnn_loss.backward()

            rcnn_optimizer.step()

            vis_o_c_loss += roi_cls_loss.item()
            vis_o_l_loss += roi_loc_loss.item()

            if (i + 1) % 40 == 0:
                vis_loss_value = [vis_o_c_loss, vis_o_l_loss]
                draw_loss_cruve(loss_name, 2501 * (epoch - 1) + i, vis_loss_value)
                vis_o_l_loss, vis_o_c_loss = 0, 0

        if (epoch + 1) % 10 == 0:
            for parameter_group in rcnn_optimizer.param_groups:
                parameter_group['lr'] *= 0.1


        svae_path = 'data/rcnn/rcnn_epoch_{}.params'.format(epoch)
        state_dict = {'rcnn': rcnn.state_dict()}
        torch.save(state_dict, svae_path)


if __name__ == '__main__':
    #train_rpn()
    #train_rcnn()
    train()