import copy
import random
import json
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset

from dataset.ImageAugmentation import (aug_croppad, aug_flip, aug_rotate, aug_scale)
from dataset.representation_infer import generate_heatmap, generate_paf


class SyntheticDataset(Dataset):
    def __init__(self, cfg, stage, transform=None, with_augmentation=False, with_mds=False):

        self.transform = transform

        self.train_data = list()
        self.val_data = list()
        DATASET = cfg.dataset

        with open(DATASET.Infer_JSON_PATH) as data_file:
            data_this = json.load(data_file)
            data = data_this['root']

        for i in range(len(data)):
            if data[i]['isValidation'] != 0:
                self.val_data.append(data[i])
            else:
                self.train_data.append(data[i])

        self.input_shape = DATASET.INPUT_SHAPE
        self.output_shape = DATASET.OUTPUT_SHAPE
        self.stride = DATASET.STRIDE


        # keypoints information
        self.root_idx = DATASET.ROOT_IDX
        self.keypoint_num = DATASET.KEYPOINT.NUM
        self.gaussian_kernels = DATASET.TRAIN.GAUSSIAN_KERNELS
        self.paf_num = DATASET.PAF.NUM
        self.paf_vector = DATASET.PAF.VECTOR
        self.paf_thre = DATASET.PAF.LINE_WIDTH_THRE

        # augmentation information
        self.with_augmentation = with_augmentation
        self.params_transform = dict()
        self.params_transform['crop_size_x'] = DATASET.INPUT_SHAPE[1]
        self.params_transform['crop_size_y'] = DATASET.INPUT_SHAPE[0]
        self.params_transform['center_perterb_max'] = DATASET.TRAIN.CENTER_TRANS_MAX
        self.params_transform['max_rotate_degree'] = DATASET.TRAIN.ROTATE_MAX
        self.params_transform['flip_prob'] = DATASET.TRAIN.FLIP_PROB
        self.params_transform['flip_order'] = DATASET.KEYPOINT.FLIP_ORDER
        self.params_transform['stride'] = DATASET.STRIDE
        self.params_transform['scale_max'] = DATASET.TRAIN.SCALE_MAX
        self.params_transform['scale_min'] = DATASET.TRAIN.SCALE_MIN

        self.with_mds = with_mds
        self.max_people = cfg.DATASET.MAX_PEOPLE

    def __len__(self):
        return len(self.train_data)

    def get_anno(self, meta_data):
        anno = dict()
        anno['dataset'] = meta_data['dataset'].upper()
        anno['img_height'] = int(meta_data['img_height'])
        anno['img_width'] = int(meta_data['img_width'])

        anno['isValidation'] = meta_data['isValidation']
        anno['bodys'] = np.asarray(meta_data['bodys'])
        anno['center'] = np.array([anno['img_width']//2, anno['img_height']//2])
        return anno

    def remove_illegal_joint(self, meta):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        for i in range(len(meta['bodys'])):
            mask_ = np.logical_or.reduce((meta['bodys'][i][:, 0] >= crop_x,
                                          meta['bodys'][i][:, 0] < 0,
                                          meta['bodys'][i][:, 1] >= crop_y,
                                          meta['bodys'][i][:, 1] < 0))

            meta['bodys'][i][mask_ == True, 3] = 0
        return meta

    def __getitem__(self, index):
        data = copy.deepcopy(self.train_data[index])

        meta_data = self.get_anno(data)

        img = np.ones((meta_data['img_height'],meta_data['img_width'],3))
        
        if self.with_augmentation:
            meta_data, img = aug_rotate(meta_data, img, self.params_transform)
        else:
            self.params_transform['center_perterb_max'] = 0

        meta_data, img = aug_croppad(meta_data, img, self.params_transform, False)

        if self.with_augmentation:
            meta_data, img = aug_flip(meta_data, img, self.params_transform)

        meta_data = self.remove_illegal_joint(meta_data)

        labels_num = len(self.gaussian_kernels)

        labels = np.zeros((labels_num, self.keypoint_num + self.paf_num*3, *self.output_shape))
        heatmaps = np.zeros((self.keypoint_num + self.paf_num*3, *self.output_shape))
        valid = np.ones((self.keypoint_num + self.paf_num*3, 1), np.float)

        noise_bodys = meta_data['bodys'].copy()

        # generate heatmaps
        kernel_size = random.randint(0,labels_num - 1)
        # heatmaps
        heatmaps[:self.keypoint_num] = generate_heatmap(noise_bodys, self.output_shape, self.stride, \
                    self.keypoint_num, kernel=self.gaussian_kernels[kernel_size], training_flag=True)
        # pafs + relative depth
        heatmaps[self.keypoint_num:] = generate_paf(noise_bodys, self.output_shape, self.params_transform, \
                    self.paf_num, self.paf_vector, max(1, (3-kernel_size))*self.paf_thre, self.with_mds, training_flag=True)
        
        # generate labels
        for i in range(labels_num):
            # heatmaps
            labels[i][:self.keypoint_num] = generate_heatmap(meta_data['bodys'], self.output_shape, self.stride, \
                        self.keypoint_num, kernel=self.gaussian_kernels[i], training_flag=False)
            # pafs + relative depth
            labels[i][self.keypoint_num:] = generate_paf(meta_data['bodys'], self.output_shape, self.params_transform, \
                        self.paf_num, self.paf_vector, max(1, (3-i))*self.paf_thre, self.with_mds, training_flag=False)
        

        
        heatmaps = torch.from_numpy(heatmaps).float()
        labels = torch.from_numpy(labels).float()
        valid = torch.from_numpy(valid).float()


        return heatmaps, valid, labels








