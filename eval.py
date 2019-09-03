import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from dataset import custom_dataset
from torch.utils.data import DataLoader

from eval_tool import calc_map
from nms import nms
from rcnn import RCNN
from rcnn_target_creator import RCNNTargetCreator
from resnet import resnet101
from rpn import RPN
from tools import loc2bbox, at, bbox_iou


def test():
    cuda = True

    test_dataset = custom_dataset(split='test')
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #resnet = Resnet101().eval()
    resnet = resnet101()
    rpn = RPN()
    rcnn = RCNN()
    if cuda:
        resnet = resnet.cuda()
        rpn = rpn.cuda()
        rcnn = rcnn.cuda()

    rpn_check_point = torch.load('/home/licheng/home/licheng/projects/cnet/data/cnet.model.state.19.pkl')
    rpn.load_state_dict(rpn_check_point['rpn'])
    resnet.load_state_dict(rpn_check_point['resnet'])

    rcnn_check_point = torch.load("/home/licheng/home/licheng/projects/cnet/data/rcnn/rcnn_epoch_19.params")
    rcnn.load_state_dict(rcnn_check_point['rcnn'])
    """
    rpn_check_point = torch.load('/home/licheng/home/licheng/projects/cnet/data/rpn/rpn_epoch_19.params')
    #resnet.load_state_dict(check_point['resnet'])
    rpn.load_state_dict(rpn_check_point['rpn'])
    #resnet.load_state_dict(check_point['resnet'])
    rcnn_check_point = torch.load('/home/licheng/home/licheng/projects/cnet/data/rcnn/rcnn_epoch_16.params')
    rcnn.load_state_dict(rcnn_check_point['rcnn'])
    """
    pred_bboxes = list()
    pred_labels = list()
    pred_scores = list()

    gt_boxes = list()
    gt_labels = list()
    rcnn_target_creator = RCNNTargetCreator()
    with torch.no_grad():
        for img_batch, bndboxes_batch, labels_batch in test_data_loader:
            img, bndboxes, labels = img_batch, bndboxes_batch[0], labels_batch[0]
            if cuda:
                img, bndboxes, labels = img.cuda(), bndboxes.cuda(), labels.cuda()
            feature = resnet(img.float())
            #if cuda:
            #    feature = feature.cuda()
            rois, anchors, rpn_loc, rpn_score = rpn(feature, feature_stride=16)
            sample_roi, gt_roi_label, gt_roi_loc = rcnn_target_creator(rois, bndboxes.cpu().numpy(), labels)

            rois = at.toTensor(rois)
            roi_cls_loc, roi_score = rcnn(rois, feature)

            look_score1 = np.array(roi_score.cpu().detach())
            pred_score = F.softmax(roi_score, dim=1)

            look_score1 = np.array(pred_score.cpu().detach())
            pred_score = pred_score.cpu().detach().numpy()

            mean = torch.Tensor((0., 0., 0., 0.)).repeat(cfg.n_class)[None].cuda()
            std  = torch.Tensor((0.1, 0.1, 0.2, 0.2)).repeat(cfg.n_class)[None].cuda()
            roi_cls_loc = (roi_cls_loc * std + mean)

            roi_cls_loc = at.toTensor(roi_cls_loc)
            roi_cls_loc = roi_cls_loc.view(-1, cfg.n_class, 4)
            rois = rois.view(-1, 1, 4).expand_as(roi_cls_loc)

            # expand dim as loc
            #rois = rois.reshape(-1, 1, 4)[:, [int(x) for x in np.zeros(cfg.n_class).tolist()], :]

            #roi_cls_loc = at.toTensor(roi_cls_loc)
            #roi_cls_loc = roi_cls_loc.view(roi_cls_loc.shape[0], -1, 4)

            #pred_box = loc2bbox(at.toNumpy(rois).reshape(-1, 4), roi_cls_loc.view(-1, 4).cpu().detach().numpy())
            pred_box = loc2bbox(at.toNumpy(rois).reshape(-1, 4), roi_cls_loc.view(-1, 4).cpu().detach().numpy())

            # clip box
            pred_box[:, 0::2] = np.clip(pred_box[:, 0::2], 0, img.shape[3])
            pred_box[:, 1::2] = np.clip(pred_box[:, 1::2], 0, img.shape[2])


            gt_box = list(bndboxes_batch.cpu().numpy())
            gt_label = list(labels_batch.cpu().numpy())

            bbox = list()
            label = list()
            score = list()

            for class_index in range(1, cfg.n_class):
                each_bbox = pred_box.reshape((-1, cfg.n_class, 4))[:, class_index, :]
                each_score = pred_score[:, class_index]
                mask = each_score > cfg.pred_score_thresh
                each_bbox = each_bbox[mask]
                each_score = each_score[mask]
                keep = nms(each_bbox, each_score, cfg.pred_nms_thresh)
                bbox.append(each_bbox[keep])
                score.append(each_score[keep])
                label.append(class_index * np.ones((len(keep),)))
            bbox = np.concatenate(bbox, axis=0).astype(np.float32)
            score = np.concatenate(score, axis=0).astype(np.float32)
            label = np.concatenate(label, axis=0).astype(np.int32)
            print('gt_info:', gt_box, gt_label)
            print('sample roi', sample_roi[0])
            print('predict info:', bbox, score, label)

            pred_bboxes += [bbox]
            pred_scores += [score]
            pred_labels += [label]
            gt_boxes += gt_box
            gt_labels += gt_label

        result = calc_map(pred_bboxes, pred_labels, pred_scores, gt_boxes, gt_labels)
        print(result)



if __name__ == '__main__':
    test()
