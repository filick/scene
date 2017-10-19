import os
from functools import partial
import pickle
import torch
import torch.nn as nn
import torchvision
from .Preact_resnet50_places365 import Preact_resnet50_places365
from .resnet152_places365 import resnet152_places365

support_models = ['alexnet', 'densenet161', 'resnet18', 'resnet50', 
                  'preact_resnet50', 'resnet152', 
                  'Imag_ResNet50','Imag_ResNet152','Imag_Densenet161','Imag_Inception_v3']

model_file_root = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'places365')


def load_model(name, use_gpu=True, num_classes=80):
    if not name in support_models:
        raise ValueError("No such places365-pretrained model found")
        
    if name in ['preact_resnet50']:
        model = Preact_resnet50_places365
        model.load_state_dict(torch.load(os.path.join(model_file_root, 'Preact_resnet50_places365.pth')))
        model._modules['12']._modules['1'] = nn.Linear(2048, num_classes)
    elif name in ['resnet152']:
        model = resnet152_places365
        model.load_state_dict(torch.load(os.path.join(model_file_root, 'resnet152_places365.pth')))
        model._modules['10']._modules['1'] = nn.Linear(2048, num_classes)
        
    elif name in ['Imag_ResNet50']:
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name in ['Imag_ResNet152']:
        model = torchvision.models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name in ['Imag_Densenet161']:
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name in ['Imag_Inception_v3']:
        model = torchvision.models.inception_v3(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        model_file = os.path.join(model_file_root, 'whole_%s_places365.pth.tar' % (name))
    
        ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        if use_gpu:
            model = torch.load(model_file, pickle_module=pickle)
        else:
            # model trained in GPU could be deployed in CPU machine like this!
            model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle) 
    
        if name.startswith('resnet'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == 'alexnet':
            model.classifier._modules['6'] = nn.Linear(4096, num_classes)
        elif name.startswith('densenet'):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model