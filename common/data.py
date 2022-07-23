''' data '''
import random
import os.path as osp
import pickle
import numpy as np
import tensorflow as tf


class DIV2K(tf.keras.utils.Sequence):
    ''' DIV2K datasets '''
    def __init__(self, mode, scale, patch_size=None, batch_size=None, iters_per_batch=1000, flip=True, rot=True):
        assert mode in ['train', 'valid', 'df2k']
        self.mode = mode
        self.scale = scale
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.iters_per_batch = iters_per_batch

        if self.mode == 'df2k':
            self.img_list = [f'{i:04d}.pt' for i in range(1, 801)] + [f'{i:06d}.pt' for i in range(1, 2651)]
            self.dataroot_hr = './datasets/DF2K/DF2K_train_HR/'
            self.dataroot_lr = './datasets/DF2K/DF2K_train_LR_bicubic/'
            self.flip = flip
            self.rot = rot
        elif self.mode == 'train':
            self.img_list = [f'{i:04d}.pt' for i in range(1, 801)]
            self.dataroot_hr = './datasets/DIV2K/DIV2K_train_HR/'
            self.dataroot_lr = f'./datasets/DIV2K/DIV2K_train_LR_bicubic/X{scale}/'
            self.flip = flip
            self.rot = rot
        else:
            self.img_list = [f'{i:04d}.pt' for i in range(801, 901)]
            self.dataroot_hr = './datasets/DIV2K/DIV2K_valid_HR/'
            self.dataroot_lr = f'./datasets/DIV2K/DIV2K_valid_LR_bicubic/X{scale}/'

    def shuffle(self):
        ''' shuffle the datasets '''
        random.shuffle(self.img_list)

    def __len__(self):
        if self.mode == 'train' or self.mode == 'df2k':
            return self.iters_per_batch
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            start = (idx * self.batch_size)
            end = start + self.batch_size
            lr_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
            hr_batch = np.zeros((self.batch_size, self.patch_size * self.scale, self.patch_size * self.scale, 3), dtype=np.float32)
            for i in range(start, end):
                lr, hr = self.get_image_pair(i % len(self.img_list))
                lr_batch[i - start] = lr
                hr_batch[i - start] = hr
        else:
            lr, hr = self.get_image_pair(idx)
            lr_batch, hr_batch = np.expand_dims(lr, 0), np.expand_dims(hr, 0)

        return (lr_batch).astype(np.float32), (hr_batch).astype(np.float32)

    def get_image_pair(self, idx):
        ''' get_image_pair '''
        # load img
        hr = self.read_img(osp.join(self.dataroot_hr, self.img_list[idx]))
        lr = self.read_img(osp.join(self.dataroot_lr, self.img_list[idx]))

        if self.mode == 'train':
            lr_patch, hr_patch = self.get_pair_patch(lr, hr, self.patch_size, self.scale)
            lr, hr = self.augment(lr_patch, hr_patch, self.flip, self.rot)

        return lr, hr

    @staticmethod
    def read_img(img_path):
        ''' read img '''
        with open(img_path, 'rb') as f:
            img = pickle.load(f)
        return img

    @staticmethod
    def get_pair_patch(lr, hr, patch_size, scale):
        ''' get pair patch '''
        lr_h, lr_w = lr.shape[:2]

        lr_x = random.randint(0, lr_w - patch_size)
        lr_y = random.randint(0, lr_h - patch_size)
        hr_x, hr_y = lr_x * scale, lr_y * scale

        lr_patch = lr[lr_y:lr_y+patch_size, lr_x:lr_x+patch_size, :]
        hr_patch = hr[hr_y:hr_y+patch_size*scale, hr_x:hr_x+patch_size*scale, :]
        return lr_patch, hr_patch

    @staticmethod
    def augment(lr, hr, flip, rot):
        ''' augment the img '''
        hflip = flip and random.random() < 0.5
        vflip = flip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            lr = np.ascontiguousarray(lr[:, ::-1, :])
            hr = np.ascontiguousarray(hr[:, ::-1, :])
        if vflip:
            lr = np.ascontiguousarray(lr[::-1, :, :])
            hr = np.ascontiguousarray(hr[::-1, :, :])
        if rot90:
            lr = lr.transpose(1, 0, 2)
            hr = hr.transpose(1, 0, 2)
        return lr, hr
