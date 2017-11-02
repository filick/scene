import random
import math
from data import transforms


#################  Trainsforms ##################

img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def basic_train_transform(img_size):
    return transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.ToTensor(), 
                img_normalize])


def basic_validate_transform(img_size):
    return transforms.Compose([
                transforms.Resize(img_size),  
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                img_normalize])


def default_train_transform(img_size):
    return transforms.Compose([
                transforms.AdaptiveRandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.ToTensor(), 
                img_normalize])

def default_validate_transform(img_sizes):
    return transforms.Compose([
                transforms.BestCenterCrop(img_sizes),
                transforms.ToTensor(),
                img_normalize])


def multiscale_train_trainsforms(multisizes):
    return list(map(lambda size: default_train_transform(size), multisizes))


#################  Dataloader ##################

# grouping