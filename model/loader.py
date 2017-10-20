import os
from functools import partial
import pickle
import torch
import torch.nn as nn
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


def load_model(arch, pretrained, use_gpu=True, num_classes=80):
    if not arch in support_models[pretrained]:
        raise ValueError("No such places365 or imagenet pretrained model found")

    if pretrained == 'imagenet':
        model = torchvision.models.__dict__[arch](pretrained=True)
    elif pretrained == 'places':
        if arch == 'preact_resnet50':
            model = Preact_resnet50_places365
            model.load_state_dict(torch.load(os.path.join(model_file_root, 'Preact_resnet50_places365.pth')))
            model._modules['12']._modules['1'] = nn.Linear(2048, num_classes)
            return model
        elif arch == 'resnet152':
            model = resnet152_places365
            model.load_state_dict(torch.load(os.path.join(model_file_root, 'resnet152_places365.pth')))
            model._modules['10']._modules['1'] = nn.Linear(2048, num_classes)
            return model
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