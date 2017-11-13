from scheme import TrainScheme
from model import load_model, FCWrapper, SPPWrapper, SPPLayer
import data
from data.dataloader import MultiTransformWrapper
from data import transforms, sample
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
        arch = "resnet152"
        pretrained = "places"

        self.hyperparams['arch'] = arch
        self.hyperparams['pretrained'] = pretrained

        model = load_model(arch, pretrained, wrapper=FCWrapper(), use_gpu=torch.cuda.is_available())
        return model


    def init_loader(self):
        batch_size = 80
        img_size = 224
        num_workers = 8
        use_gpu = torch.cuda.is_available()

        self.hyperparams['batch_size'] = batch_size
        self.hyperparams['image_size'] = img_size

        train_set = data.ChallengerSceneFolder(data.TRAIN_ROOT, basic_train_transform(img_size))
        train_loader = torch.utils.data.DataLoader(
                        train_set,
                        batch_size=batch_size, shuffle=True,
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


class MultiScaleTrainScheme(BasicTrainScheme):


    @property
    def name(self):
        self.model
        return '_'.join([self.hyperparams['arch'], self.hyperparams['pretrained'], self.hyperparams['wrapper']])


    def init_model(self):
        arch = "resnet152"
        pretrained = "places"

        self.hyperparams['arch'] = arch
        self.hyperparams['pretrained'] = pretrained

        model = load_model(arch, pretrained, wrapper=SPPWrapper(spp_layer=SPPLayer(2)), use_gpu=torch.cuda.is_available())
        self.hyperparams['wrapper'] = 'sppnet2'

        return model


    def init_loader(self):
        batch_size = 100
        # multi_imgsizes = [(224, 448),] + [(288, 384), ] * 4 + [(320, 320)] * 2 + [(384, 288)] * 2 + [(448, 224)]
        multi_imgsizes = [(288, 384),] * 7 + [(384, 288),] * 2
        num_workers = 8
        use_gpu = torch.cuda.is_available()

        self.hyperparams['batch_size'] = batch_size
        self.hyperparams['multi_image_size'] = multi_imgsizes

        train_set = data.ChallengerSceneFolder(data.TRAIN_ROOT)
        train_loader = torch.utils.data.DataLoader(
                        train_set,
                        batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=use_gpu, drop_last=True)
        multiscale_transforms = multiscale_train_trainsforms(multi_imgsizes)
        train_loader = MultiTransformWrapper(train_loader, multiscale_transforms)

        validation_set = data.ChallengerSceneFolder(data.VALIDATION_ROOT, default_validate_transform(set(multi_imgsizes)))
        sampler = sample.WithSizeSampler(validation_set)
        grouping = sample.group_by_ratio(set(multi_imgsizes))
        batch_sampler = sample.GroupingBatchSampler(sampler, batch_size, grouping)
        validation_loader = torch.utils.data.DataLoader(
                        validation_set,
                        batch_sampler=batch_sampler,
                        num_workers=num_workers, pin_memory=use_gpu)
        return train_loader, validation_loader


    def init_criterion(self):
        self.hyperparams['criterion'] = 'cross-entropy loss'
        return nn.CrossEntropyLoss()


    def init_optimizer(self):
        lr = 0.01
        betas=(0.9, 0.999)
        eps=1e-08
        weight_decay=0

        self.hyperparams['optimizer'] = 'Adam'
        self.hyperparams['lr'] = lr
        self.hyperparams['betas'] = betas
        self.hyperparams['eps'] = eps
        self.hyperparams['weight_decay'] = weight_decay

        opt = optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        lr_scheduler = lrs.ReduceLROnPlateau(opt, mode='min', factor=0.382, patience=2)
        self.hyperparams['lr_schedule'] = 'ReduceLROnPlateau, factor=0.382, patience=2'

        return opt, lr_scheduler


class TwoPhaseScheme(MultiScaleTrainScheme):


    @property
    def name(self):
        self.model
        return '_'.join([self.hyperparams['arch'], self.hyperparams['pretrained'], self.hyperparams['wrapper']])


    def init_model(self):
        arch = "resnet152"
        pretrained = "places"

        self.hyperparams['arch'] = arch
        self.hyperparams['pretrained'] = pretrained

        model = load_model(arch, pretrained, wrapper=SPPWrapper(spp_layer=SPPLayer([2,])), use_gpu=torch.cuda.is_available())
        self.hyperparams['wrapper'] = 'sppnet2s'

        return model


    def init_optimizer(self):
        lr = 0.01
        momentum = 0.9
        weight_decay= 0

        self.hyperparams['optimizer'] = 'SGD'
        self.hyperparams['lr'] = lr
        self.hyperparams['momentum'] = momentum
        self.hyperparams['weight_decay'] = weight_decay

        opt = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)

        # lr_scheduler = lrs.ReduceLROnPlateau(opt, mode='min', factor=0.382, patience=2)
        lr_scheduler = lrs.LambdaLR(opt, lambda epoch: 0.5 ** (epoch // 5))
        self.hyperparams['lr_schedule'] = 'divided by 5 every 5 epoches'

        return opt, lr_scheduler