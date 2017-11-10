'''
https://github.com/moskomule/senet.pytorch
https://github.com/moskomule/senet.pytorch/issues/3 
    use ResNet's weight for "SENet's ResNet part" and use arbitrary weights for "SENet's SE blocks"
https://github.com/KaimingHe/deep-residual-networks  
    另外需要查看ResNet的通道数
'''


from torch import nn
from .resnet152_places365 import resnet152_places365
from se_module import SELayer

print_net = True
reduction = 16

se_resnet152_places365 = resnet152_places365

# 按照每个redidual module的channel数增加se通路
se_resnet152_places365._modules['10'].add_module('se',SELayer(planes * 4, reduction))




if print_net:
    print(se_resnet152_places365)