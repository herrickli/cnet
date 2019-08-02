import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from dataset import custom_dataset
from torch.utils.data import DataLoader

from nms import nms
from rcnn import RCNN
from rcnn_target_creator import RCNNTargetCreator
from resnet import resnet101
from rpn import RPN
from tools import loc2bbox, at


def test():
    test_dataset = custom_dataset(split='test')
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #resnet = Resnet101().eval()
    resnet = resnet101(pretrained=True).eval()
    rpn = RPN().eval()
    rcnn = RCNN().eval()
    check_point = torch.load('data/cnet.model.state.9.2499.pkl')
    #resnet.load_state_dict(check_point['resnet'])
    rpn.load_state_dict(check_point['rpn'])
    rcnn.load_state_dict(check_point['rcnn'])

    pred_bboxes = list()
    pred_labels = list()
    pred_scores = list()

    for img_batch, bndboxes_batch, labels_batch in test_data_loader:
        img, bndboxes, labels = img_batch, bndboxes_batch[0], labels_batch[0]
        feature = resnet(img.float())
        rois, anchors, rpn_loc, rpn_score = rpn(feature, feature_stride=32)

        roi_cls_loc, roi_score = rcnn(rois, feature)

        mean = torch.Tensor((0., 0., 0., 0.)).repeat(cfg.n_class)[None]
        std  = torch.Tensor((0.1, 0.1, 0.2, 0.2)).repeat(cfg.n_class)[None]
        roi_cls_loc = (roi_cls_loc * std + mean)


        # expand dim as loc
        rois = rois.reshape(-1, 1, 4)[:, [int(x) for x in np.zeros(cfg.n_class).tolist()], :]

        roi_cls_loc = at.toTensor(roi_cls_loc)
        roi_cls_loc = roi_cls_loc.view(roi_cls_loc.shape[0], -1, 4)


        pred_box = loc2bbox(rois.reshape(-1, 4), roi_cls_loc.view(-1, 4).cpu().detach().numpy())
        # clip box
        pred_box[:, 0::2] = np.clip(pred_box[:, 0::2], 0, img.shape[2])
        pred_box[:, 1::2] = np.clip(pred_box[:, 1::2], 0, img.shape[3])

        look_score1 = np.array(roi_score.detach())
        pred_score = F.softmax(roi_score, dim=1)

        look_score1 = np.array(pred_score.detach())
        pred_score = pred_score.detach()

        bbox = list()
        label = list()
        score = list()

        for class_index in range(1, cfg.n_class):
            each_bbox = pred_box.reshape((-1, cfg.n_class, 4))[:, class_index, :]
            each_score = pred_score[:, class_index]
            mask = each_score > cfg.pred_score_thresh
            each_bbox = each_bbox[mask]
            each_score = each_score[mask]
            keep = nms(each_bbox, each_score.numpy(), cfg.pred_nms_thresh)
            bbox.append(each_bbox[keep])
            score.append(each_score[keep])
            label.append(class_index * np.ones((len(keep),)))
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)

        print('gt_info:', bndboxes, labels)
        print('pred_info', bbox, score, label)

        pred_bboxes.append(bbox)
        pred_scores.append(score)
        pred_labels.append(labels)

if __name__ == '__main__':
    test()
