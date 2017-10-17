import os
from functools import partial
import pickle
import torch
import torch.nn as nn


support_models = ['alexnet', 'densenet161', 'resnet18', 'resnet50']
model_file_root = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'places365')


def load_model(name, use_gpu=True, num_classes=80):
    if not name in support_models:
        raise ValueError("No such places365-pretrained model found")
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
        raise RuntimeError('The adaption of the pretrained AlexNet to the current task has not been implemented.')
    elif name.startswith('densenet'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model