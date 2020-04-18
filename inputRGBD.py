from torch.utils.data.dataset import Dataset, ConcatDataset
from torch.utils.data.dataloader import DataLoader
from skimage import io
import math
from math import log
import torch
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image
import numpy as np


class RGBDDataset(Dataset):
    def __init__(self, scene_name='', path='', seq='',
                 frame_num=0):
        self.scene_name = scene_name
        self.path = path
        self.seq = seq
        self.frame_num = frame_num

    def __getitem__(self, index):
        if index >= self.frame_num:  # if user input a index larger than max
            raise IndexError
        if index != 0:
            prefix_num = 5 - math.floor(log(index, 10))
        else:
            prefix_num = 5
        temp = self.path + self.scene_name + '/' + self.seq + '/frame-'
        for i in range(0, prefix_num):
            temp += '0'
        this_image_color_path = temp + str(index) + '.color.png'
        this_image_depth_path = temp + str(index) + '.depth.png'
        this_frame_pose_path = temp + str(index) + '.pose.txt'
        # print(this_image_color_path)
        # print(this_image_depth_path)
        # print(this_frame_pose_path)
        img_color = np.array(Image.open(this_image_color_path))
        img_depth = np.array(Image.open(this_image_depth_path))
        frame_pose = np.loadtxt(this_frame_pose_path)  # , dtype='float64')
        # print(img_color)

        img_color = torch.from_numpy(img_color)
        img_depth = torch.from_numpy(img_depth)
        frame_pose = torch.from_numpy(frame_pose)
        result = (index, img_color, img_depth, frame_pose)
        return result

    def __len__(self):
        return self.frame_num

    def __add__(self, other):
        return ConcatDataset([self, other])


# new_dataset = RGBDDataset('office', './', 'seq-01', 1000)
# index, img_color, img_depth, frame_pose = new_dataset.__getitem__(0)
# print(frame_pose)
# dataLoader = DataLoader(new_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
# for step, (index, img_color, img_depth, frame_pose) in enumerate(dataLoader):
#     print(step, index, img_color, img_depth, frame_pose)
# image_color_path = './office/seq-01/seq-01/'
# index = 200
# prefix_num = 6 - math.ceil(log(index, 10))
# print(prefix_num)
# this_path = image_color_path + 'frame-'
# for i in range(0, prefix_num):
#     this_path += '0'
# this_path += str(index) + '.color.png'
# print(this_path)
# img_skimage_color = io.imread('./office/seq-01/seq-01/frame-000000.color.png')
# img_skimage_depth = io.imread('./office/seq-01/seq-01/frame-000000.depth.png')
# print('The shape of \n img_skimage_color is {} \n image_skimage_depth is {}'.format(img_skimage_color.shape, img_skimage_depth.shape))
# print(img_skimage_depth)
