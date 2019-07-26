import os
import pickle
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import cv2
from config import cfg


class custom_dataset(Dataset):
    def __init__(self, data_root='/home/cheng/data/custom/'):
        self.data_root = data_root
        self.pkl_data = None
        self._pickle_data(split='train')

    def __getitem__(self, index):
        with open(self.pkl_data, 'rb') as f:
            data = pickle.load(f)
        #pprint(data)
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
            labels.append(cfg.dataset_class.index(object['name']) + 1)
            bndboxes.append(object['bndbox'])
        bndboxes = np.stack(bndboxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        return img, bndboxes, labels

    def __len__(self):
        with open(self.pkl_data, 'rb') as f:
            data = pickle.load(f)
        return len(data)

    def _pickle_data(self, split):
        """
        one picture ground truth data organized in this format
        {'path':<anno_path>,
         'objects':[{'name:'<obj_name>,'bndbox':[xmin,ymin,xmax,ymax], ...}]
         }
        :param split:
        :return:
        """
        if not os.path.exists('data/gt_info.pkl'):
            print('Writing pkl file...')
            data = list()
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
            self.pkl_data = 'data/gt_info.pkl'
            with open(self.pkl_data, 'wb') as pkl_file:
                pickle.dump(data, pkl_file)
        else:
            self.pkl_data = 'data/gt_info.pkl'


if __name__ == '__main__':
    dataset = custom_dataset('/home/cheng/data/custom/')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    data_iter = iter(data_loader)
    """ 
    for img, bndboxes, labels in data_loader:
        print(img)
    """
    print(next(data_iter))
   # print(np.stack(dataset[2][0]))
