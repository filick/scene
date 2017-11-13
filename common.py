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



#################  Functions  ##################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res