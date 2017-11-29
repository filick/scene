'''
ref
https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3

https://discuss.pytorch.org/t/how-could-i-build-network-structure-equivelant-to-old-nn-concat-or-nn-parallel/686/5
https://discuss.pytorch.org/t/nn-module-with-multiple-inputs/237/3
https://discuss.pytorch.org/t/how-to-train-an-ensemble-of-two-cnns-and-a-classifier/3026/7
https://github.com/YanWang2014/PyTorchText/blob/master/models/MultiModelAll2.py
'''

import torch
import torch.nn as nn
import torchvision
from .resnet152_places365 import resnet152_places365
from .spp_layer import SPPLayer

test_model = True

arch = ['resnet152' ,'resnet152]']
pretrained = ['imagenet', 'places']
best_check = []

for i in range(len(arch)):
    checkpoint_filename = arch[i] + '_' + pretrained[i]
    best_check.append('../checkpoint/' + checkpoint_filename + '_best.pth.tar')
    
num_levels = 1 # 1 = fcn
pool_type = 'avg_pool'



model1 = torchvision.models.resnet152(pretrained=False)
if not test_model:
    checkpoint = torch.load(best_check[0])
    model1.load_state_dict(checkpoint['state_dict'])
model1.avgpool = SPPLayer(num_levels, pool_type)
new_classifier = nn.Sequential(*list(model1.fc.children())[:-1])
model1.fc = new_classifier

model2 = resnet152_places365
if not test_model:
    checkpoint = torch.load(best_check[1])
    model2.load_state_dict(checkpoint['state_dict'])
model2._modules['8'] = SPPLayer(num_levels, pool_type)
new_classifier = nn.Sequential(*list(model2._modules['10']._modules['1'] .children())[:-1])
model2._modules['10']._modules['1']  = new_classifier


class Two_path(nn.Module):
    def __init__(self, num_classes):
        super(Two_path, self).__init__()
        self.path1 = model1
        self.path2 = model2
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        in1 = self.path1(x).view(x.size(0),-1)
        in2 = self.path2(x).view(x.size(0),-1)
        x = torch.cat((in1,in2),1)
        x = self.fc(x)
        return x





if test_model:
    import torch
    from torch.autograd import Variable
    x = Variable(torch.ones(2, 2, 2), requires_grad=True)
    y = Variable(torch.zeros(2, 2, 2), requires_grad=True)
    print(x.view(x.size(0),-1).size())
    print(torch.stack((x.view(x.size(0),-1),y.view(y.size(0),-1))),1)
    
    x1 = x.view(x.size(0),-1)
    x = Variable(torch.zeros(2, 2, 2), requires_grad=True)
    y1 = y.view(y.size(0),-1)
    print(torch.cat((x1,y1),1))