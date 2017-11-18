import os
import math
from functools import partial
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .Preact_resnet50_places365 import Preact_resnet50_places365
from .resnet152_places365 import resnet152_places365
import torchvision.models


support_models = {
    'places': ('alexnet', 'densenet161', 'resnet18', 'resnet50', 'preact_resnet50', 'resnet152'),
    'imagenet': tuple(filter(lambda x: (x.lower() == x) and (not x.startswith('_') and \
                                       (not x in ['densenet', 'resnet', 'vgg', 'inception', 'squeezenet'])) , \
                             dir(torchvision.models)))
}

model_file_root = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'places365')


def load_model(arch, pretrained, use_gpu=True, wrapper=None):
    if not arch in support_models[pretrained]:
        raise ValueError("No such places365 or imagenet pretrained model found")

    if pretrained == 'imagenet':
        model = torchvision.models.__dict__[arch](pretrained=True)
    elif pretrained == 'places':
        if arch == 'preact_resnet50':
            model = Preact_resnet50_places365
            model.load_state_dict(torch.load(os.path.join(model_file_root, 'Preact_resnet50_places365.pth')))
        elif arch == 'resnet152':
            model = resnet152_places365
            model.load_state_dict(torch.load(os.path.join(model_file_root, 'resnet152_places365.pth')))
        else:
            model_file = os.path.join(model_file_root, 'whole_%s_places365.pth.tar' % (arch))
            ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            if use_gpu:
                model = torch.load(model_file, pickle_module=pickle)
            else:
                # model trained in GPU could be deployed in CPU machine like this!
                model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle) 
    if wrapper is not None:
        model = wrapper(model, arch, pretrained)
    return model


class ModelWrapper(object):
    
    def __call__(self, model, arch, pretrained):
        raise NotImplementedError("ModelWrapper")


class FCWrapper(ModelWrapper):

    def __init__(self, num_classes=80):
        self.num_classes = num_classes

    def __call__(self, model, arch, pretrained):
        num_classes = self.num_classes
        if pretrained == 'places':
            if arch == 'preact_resnet50':
                model._modules['12']._modules['1'] = nn.Linear(2048, num_classes)
                return model
            elif arch == 'resnet152':
                model._modules['10']._modules['1'] = nn.Linear(2048, num_classes)
                return model

        if arch.startswith('resnet'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch.startswith('densenet'):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif arch.startswith('inception'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch.startswith('vgg') or arch == 'alexnet':
            model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features, num_classes)
        else:
            raise NotImplementedError('This pretrained model has not been adapted to the current tast yet.')
        return model


class AdaptiveAvgPoolWrapper(FCWrapper):

    def __call__(self, model, arch, pretrained):
        if not arch.startswith('resnet'):
            raise NotImplementedError("Currently AdaptiveAvgPool only supports resnets")

        model = super(AdaptiveAvgPoolWrapper, self).__call__(model, arch, pretrained)
        if pretrained == 'places':
            if arch == 'preact_resnet50':
                model._modules['10'] = nn.AdaptiveAvgPool2d(1)
                return model
            elif arch == 'resnet152':
                model._modules['8'] = nn.AdaptiveAvgPool2d(1)
                return model
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        return model


class SPPWrapper(ModelWrapper):

    def __init__(self, num_classes=80, spp_layer=None):
        self.num_classes = num_classes
        if spp_layer == None:
            spp_layer = SPPLayer(3)
        self.spp_layer = spp_layer


    def __call__(self, model, arch, pretrained):
        if not arch.startswith('resnet'):
            raise NotImplementedError("Currently AdaptiveAvgPool only supports resnets")

        num_classes = self.num_classes
        if pretrained == 'places':
            if arch == 'preact_resnet50':
                model._modules['10'] = self.spp_layer
                model._modules['12']._modules['1'] = nn.Linear(2048 * self.spp_layer.outscale, num_classes)
                return model
            elif arch == 'resnet152':
                model._modules['8'] = self.spp_layer
                model._modules['10']._modules['1'] = nn.Linear(2048 * self.spp_layer.outscale, num_classes)
                return model

        model.avgpool = self.spp_layer
        model.fc = nn.Linear(model.fc.in_features * self.spp_layer.outscale, num_classes)
        return model



class SPPLayer(nn.Module):


    def __init__(self, levels, pool_type='avg_pool'):
        super(SPPLayer, self).__init__()

        if isinstance(levels, int):
            self.levels = [2 ** l for l in range(levels)]
        elif isinstance(levels, (list, tuple)):
            self.levels = levels
        if isinstance(pool_type, str):
            self.pool_type = [pool_type] * len(self.levels)
        elif isinstance(pool_type, (list, tuple)):
            self.pool_type = pool_type
        if len(self.levels) != len(self.pool_type):
            raise ValueError("The number of level must be equal to the number of pool_type")
        self.outscale = sum([l * l for l in self.levels])


    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for level, pool_type in zip(self.levels, self.pool_type):
            if pool_type == 'max_pool':
                pooling_layers.append(F.adaptive_max_pool2d(x, level).view(bs, c, -1))
            elif pool_type == 'avg_pool':
                pooling_layers.append(F.adaptive_avg_pool2d(x, level).view(bs, c, -1))
            else:
                raise ValueError("Pool_type should be one in ('avg_pool', 'max_pool')")
        x = torch.cat(pooling_layers, dim=-1)
        return x






