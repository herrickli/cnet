import os
import pickle
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import cv2
from skimage.transform import resize
from torchvision.transforms import Normalize

from config import cfg



class custom_dataset(Dataset):
    """
    the data in directory should be organized like this
    - data
      - custom
        - train.txt # only prefix, no postfix
        - test.txt # same as above
        - Annotations/your_annotation_file.xml
        - Images/your_image_file.jpg
    """
    def __init__(self, split='train', data_root='/home/licheng/data/custom/', augment=False):
        self.data_root = data_root
        self.split = split
        self.augment = augment
        self.pkl_data = None
        self.min_size = 600
        self.max_size = 1000
        self._pickle_data(split=self.split)

    def __getitem__(self, index):
        with open(self.pkl_data, 'rb') as f:
            if self.split == 'train':
                head_info = pickle.load(f)
            data = pickle.load(f)
        xml_path = data[index]['path']
        print(xml_path)
        img_path = os.path.join(self.data_root, 'Images', xml_path.split('/')[-1].split('.')[0] + '.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # bgr, (h,w)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        objects = data[index]['objects']
        bndboxes = list()
        labels = list()
        for object in objects:
            labels.append(cfg.dataset_class.index(object['name']))
            bndboxes.append(object['bndbox'])
        bndboxes = np.stack(bndboxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)

        _ori_channel, ori_h, ori_w = img.shape
        scale1 = self.min_size / min(ori_h, ori_w)
        scale2 = self.max_size / max(ori_h, ori_w)
        scale = min(scale1, scale2)
        img = img / 255
        img = resize(img, (_ori_channel, ori_h * scale, ori_w * scale), mode='reflect', anti_aliasing=False)
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(torch.from_numpy(img).float())
        img = img.numpy()

        bbox = bndboxes.copy()
        bbox = bbox * scale

        return img, bbox, labels

    def __len__(self):
        with open(self.pkl_data, 'rb') as f:
            if self.split == 'train':
                head_info = pickle.load(f)
            data = pickle.load(f)
        return len(data)

    def _pickle_data(self, split):
        if split == 'train':
            if not os.path.exists('data/gt_train_info.pkl'):
                self._write_pickle_file(split=split)
            else:
                self.pkl_data = 'data/gt_train_info.pkl'
                with open(self.pkl_data, 'rb') as f:
                    head_info = pickle.load(f)
                if head_info['augment'] != str(self.augment):
                    print('pkl file confilct, rewriting...')
                    self._write_pickle_file(split=split)
        elif split == 'test':
            if not os.path.exists('data/gt_test_info.pkl'):
                self._write_pickle_file(split=split)
            else:
                self.pkl_data = 'data/gt_test_info.pkl'
        else:
            raise ValueError('`split` argument is not wright.')

    def _write_pickle_file(self, split):
        """
        one picture ground truth data organized in this format
        {'path':<anno_path>,
         'objects':[{'name:'<obj_name>,'bndbox':[xmin,ymin,xmax,ymax], ...}]
         }
        :param split:
        :return:
        """

        print('Writing pkl file...')
        data = list()
        if split == 'train':
            head_info = dict()
            head_info['augment'] = str(self.augment)
        with open(self.data_root + split + '.txt', 'r') as f:
            for line in f.readlines():
                gt_info = dict()
                objects = list()
                anno_path = os.path.join(self.data_root, 'Annotations/') + line.strip() + '.xml'
                gt_info['path'] = anno_path
                anno = ET.parse(anno_path)
                for object in anno.findall('object'):
                    obj = dict()
                    obj['name'] = object.find('name').text
                    obj['bndbox'] = list()
                    for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
                        obj['bndbox'].append(float(object.find('bndbox').find(tag).text))
                    objects.append(obj)
                gt_info['objects'] = objects
                data.append(gt_info)
        if split == 'train':
            self.pkl_data = 'data/gt_train_info.pkl'
        else:
            self.pkl_data = 'data/gt_test_info.pkl'
        with open(self.pkl_data, 'wb') as pkl_file:
            if split == 'train':
                pickle.dump(head_info, pkl_file)
            pickle.dump(data, pkl_file)


if __name__ == '__main__':

    dataset = custom_dataset(data_root='/home/cheng/data/custom/')
    print(len(dataset))
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    data_iter = iter(data_loader)
    """
    for img, bndboxes, labels in data_loader:
        print(img.shape)
        print(bndboxes)
        print(labels)
    
    print(next(data_iter))

    with open('data/gt_info.pkl', 'rb') as f:
        data = pickle.load(f)
        data2 = pickle.load(f)

    pprint(data)
    print(data2)
   # print(np.stack(dataset[2][0]))
    """