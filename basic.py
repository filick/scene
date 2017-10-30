from scheme import TrainScheme
from model import load_model
import data
from data import transforms
import torch.utils.data.dataloader
import torch.cuda
from common import *
import torch.nn as nn
from torch import optim
import torch.optim.lr_scheduler as lrs


class BasicTrainScheme(TrainScheme):


    @property
    def name(self):
        self.model
        return self.hyperparams['arch'] + '_' + self.hyperparams['pretrained']


    def init_model(self):
        arch = "resnet50"
        pretrained = "places"

        self.hyperparams['arch'] = arch
        self.hyperparams['pretrained'] = pretrained

        model = load_model(arch, pretrained, use_gpu=torch.cuda.is_available())
        return model


    def init_loader(self):
        batch_size = 128
        img_size = 224
        num_workers = 8
        use_gpu = torch.cuda.is_available()

        self.hyperparams['batch_size'] = batch_size
        self.hyperparams['image_size'] = img_size

        train_set = data.ChallengerSceneFolder(data.TRAIN_ROOT, basic_train_transform(img_size))
        train_loader = torch.utils.data.DataLoader(
                        train_set,
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=use_gpu)

        validation_set = data.ChallengerSceneFolder(data.VALIDATION_ROOT, basic_validate_transform(img_size))
        validation_loader = torch.utils.data.DataLoader(
                        validation_set,
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=use_gpu)
        return train_loader, validation_loader


    def init_criterion(self):
        self.hyperparams['criterion'] = 'cross-entropy loss'
        return nn.CrossEntropyLoss()


    def init_optimizer(self):
        lr = 0.0001
        betas=(0.9, 0.999)
        eps=1e-08
        weight_decay=0

        self.hyperparams['optimizer'] = 'Adam'
        self.hyperparams['lr'] = lr
        self.hyperparams['betas'] = betas
        self.hyperparams['eps'] = eps
        self.hyperparams['weight_decay'] = weight_decay

        opt = optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        lr_scheduler = lrs.LambdaLR(opt, lambda epoch: 0.2 ** (epoch // 5))
        self.hyperparams['lr_schedule'] = 'divided by 5 every 5 epoches'

        return opt, lr_scheduler