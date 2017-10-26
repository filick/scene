# https://github.com/pytorch/vision/blob/master/torchvision/transforms.py

import torch
from torchvision import transforms
import random
from PIL import Image
from .transforms_master import ColorJitter, scale, ten_crop

input_size = 224 # currenttly fixed
train_scale = 256 # currently not used in 'train'
test_scale = 256
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def my_transform(img):
    img = scale(img, test_scale)
    imgs = ten_crop(img, input_size)  # this is a list of PIL Images
    return torch.stack([normalize(transforms.to_tensor(x)) for x in imgs], 0) # returns a 4D tensor

# following ResNet paper, note that center crop should be removed if we can handle different image sizes in a batch
def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
class HorizontalFlip(object):
    def __init__(self, flip_flag):
        self.flip_flag = flip_flag
    def __call__(self, img):
        if self.flip_flag:
            return hflip(img)
        else:
            return img
def my_transform_multiscale_test(varied_scale, flip_flag):  
    return transforms.Compose([
        transforms.Scale(varied_scale),  
        transforms.CenterCrop(varied_scale), 
        HorizontalFlip(flip_flag),
        transforms.ToTensor(),
        normalize
    ])


composed_data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(input_size), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        normalize
    ]),
    'multi_scale_train': transforms.Compose([   ## following ResNet paper, but not include the standard color augmentation from AlexNet
        transforms.Scale(random.randint(256, 512)),  # May be adjusted to be bigger
        transforms.RandomCrop(input_size),  # not RandomSizedCrop
        transforms.RandomHorizontalFlip(), 
        ColorJitter(), # different from AlexNet's PCA method which is adopted in the ResNet paper?
        transforms.ToTensor(), 
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Scale(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Scale(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'ten_crop': my_transform
}


def data_transforms(phase):
    return composed_data_transforms[phase]
