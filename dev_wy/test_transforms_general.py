#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 22:33:47 2017

@author: wayne
- 训练时数据增强
    https://github.com/pytorch/vision/blob/master/torchvision/transforms.py （目前没有旋转？！）
    https://github.com/ncullen93/torchsample
        https://github.com/ncullen93/torchsample/blob/master/examples/Transforms%20with%20Pytorch%20and%20Torchsample.ipynb
        https://github.com/ncullen93/torchsample/issues/15
    http://augmentor.readthedocs.io/en/master/userguide/mainfeatures.html#

torchvision
def adjust_gamma(img, gamma, gain=1):
    Perform gamma correction on an image.
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
        I_out = 255 * gain * ((I_in / 255) ** gamma)
    See https://en.wikipedia.org/wiki/Gamma_correction for more details.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
def five_crop(img, size):
    Crop the given PIL Image into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
        tuple: tuple (tl, tr, bl, br, center) corresponding top left,
            top right, bottom left, bottom right and center crop.
ten_crop
class LinearTransformation(object):
    Transform a tensor image with a square transformation matrix computed
    offline.
class ColorJitter(object):
    Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.

torchsample
affine transformations
其他一些不太常见的方法
"""

import os
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchsample

batch_size = 2

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(), #Horizontally flip the given PIL.Image randomly with a probability of 0.5.
        transforms.ToTensor(), #Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        torchsample.transforms.Rotate(30),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #channel = (channel - mean) / std
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),  #def resize(img, size, interpolation=Image.BILINEAR):
        #transforms.CenterCrop(224), #Crops the given PIL.Image at the center.
        transforms.TenCrop(224), #编译安装master版本的vision
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


'''
load and transform data
'''
with open('../ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json', 'r') as f: #label文件
    label_raw_train = json.load(f)
with open('../ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json', 'r') as f: #label文件
    label_raw_val = json.load(f)


class SceneDataset(Dataset):

    def __init__(self, json_labels, root_dir, transform=None):
        """
        Args:
            json_labesl (list):read from official json file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_raw = json_labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_raw)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.label_raw[idx]['image_id'])
        image = Image.open(img_name)
        label = int(self.label_raw[idx]['label_id'])

        if self.transform:
            image = self.transform(image)

        return image, label


transformed_dataset_train = SceneDataset(json_labels=label_raw_train,
                                    root_dir='../ai_challenger_scene_train_20170904/scene_train_images_20170904',
                                           transform=data_transforms['train']
                                           )      
transformed_dataset_val = SceneDataset(json_labels=label_raw_val,
                                    root_dir='../ai_challenger_scene_validation_20170908/scene_validation_images_20170908',
                                           transform=data_transforms['val']
                                           )         
dataloader = {'train':DataLoader(transformed_dataset_train, batch_size=batch_size,shuffle=True, num_workers=8),
             'val':DataLoader(transformed_dataset_val, batch_size=batch_size,shuffle=False, num_workers=8)
             }
dataset_sizes = {'train': len(label_raw_train), 'val':len(label_raw_val)}


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloader['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
imshow(out)


# Get a batch of training data
inputs2, classes2 = next(iter(dataloader['val']))
print(len(inputs2))
# Make a grid from batch
#out2 = torchvision.utils.make_grid(inputs2)
#imshow(out2)