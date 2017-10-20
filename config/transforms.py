# https://github.com/pytorch/vision/blob/master/torchvision/transforms.py

import torch
from torchvision import transforms

input_size = 224
train_scale = 256 # currently not used in 'train'
test_scale = 256
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def my_transform(img):
    # do any transforms you want here
    img = transforms.resize(img, test_scale)
    imgs = transforms.ten_crop(img, input_size)  # this is a list of PIL Images
    return torch.stack([normalize(transforms.to_tensor(x)) for x in imgs], 0) # returns a 4D tensor

composed_data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(input_size), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Resize(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(test_scale),  
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'ten_crop': my_transform
}


def data_transforms(phase):
    return composed_data_transforms[phase]