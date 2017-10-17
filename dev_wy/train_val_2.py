'''
CHANGES:
- imagenet cnns: http://pytorch.org/docs/master/torchvision/models.html
- places 365 cnns: resnet 18, 50: https://github.com/CSAILVision/places365
- top3 accuracy: https://github.com/pytorch/examples/blob/master/imagenet/main.py
- 训练-验证流程: http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#
- 验证-测试流程
- 训练时数据增强-test_transforms_general
    https://github.com/pytorch/vision/blob/master/torchvision/transforms.py （目前没有旋转和各种仿射变换）
    https://github.com/ncullen93/torchsample  (有旋转等仿射变换)
        https://github.com/ncullen93/torchsample/blob/master/examples/Transforms%20with%20Pytorch%20and%20Torchsample.ipynb
        https://github.com/ncullen93/torchsample/issues/15
    http://augmentor.readthedocs.io/en/master/userguide/mainfeatures.html#
- 数据不平衡(在pytorch中很简单，用tfrecord的话估计又是一个大坑)
    https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    https://discuss.pytorch.org/t/multilabel-classification-under-unbalanced-class-distributions/2950/3
    https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/main_pytorch.py#L103
    ######### 但是目前似乎不能保证每个图片都被用到, 还有目前使用sampler的话shuffle只能是False (weightedrandomsampler: replacement=True)
        Relay Backpropagation for Effective Learning of Deep Convolutional Neural Networks    
        Learning Deep Convolutional Neural Network for Places2 Scene Recognition
        ######### 快有ClassAwareSampler了: https://github.com/pytorch/pytorch/pull/3062
- 测试时数据增强（TTA / Test Time Augmentation）: 见val_test_1_aug2,目前以five_crop和ten_crop为例(注意目前pytorch没有支持这个，借用Torchsample的transform来实现)
    slim: using multiple crops at multiple scales
    两个冠军方案都用这个了，方式不同

TODO:
- 开始跑程序，测试gpu调度，网络架构(resnet 18, 50)，学习率，数据增强。看看结果能干到0.95吗
    https://github.com/pytorch/pytorch/issues/1355
- 读懂ResNet源码

    
后续1
- ######### 读ResNet论文和源码: 恒等映射identity mappings provide reasonable preconditioning
    0Deep Residual Learning for Image Recognition
    0Identity Mappings in Deep Residual Networks
    resnext
        http://www.360doc.com/content/17/0214/19/36492363_628998225.shtml
        0Aggregated Residual Transformations for Deep Neural Networks (分组，减少参数变相增大模型)
- ######### 多尺度等
    Resnet原论文 https://arxiv.org/pdf/1512.03385.pdf
        scale augmentation: 256, 384. scale jittering: [256, 512]
            PyTorch的RandomSizedCrop可能一定程度上与这个类似
        horizontal flip
        per-pixel mean subtracted (centered raw RGB values)
            现在用的transforms.Normalize参数是imagenet训练集上的
        ### standard color augmentation: pca
            一个实现: https://github.com/mratsim/Amazon_Forest_Computer_Vision/blob/master/src/p_data_augmentation.py
            LinearTransformation? https://github.com/pytorch/vision/issues/245
            ColorJitter
        mini-batch size of 256
        weight decay of 0.0001
        ######### 10-crop testing, fullyconvolutional form as in [41, 13], and average the scores at multiple scales (images are resized such that the shorter side is in {224, 256, 384, 480, 640})
            ######### 实现?
            ### https://github.com/pytorch/vision/issues/61 比我的val_test_1_aug3简洁多了，是把crop拼进batch
    ######### resnet 用320x320训练?
        ######### 全局池化
            https://discuss.pytorch.org/t/how-do-i-feed-higher-res-images-to-model-zoo-models/1442/6
        def get_input_size(lastout=8):
            input_size = 2*lastout + 1 # 17
            input_size = 2*input_size + 1 # 35
            input_size = 2*input_size + 1 # 71
            input_size = input_size + 2 # 73
            input_size = 2*input_size + 1 # 147
            input_size = input_size + 2 # 149
            input_size = 2*input_size + 1 # 299
            return input_size #299, 这是inception系列模型的？
            
            
- dilation convolution
- attention
    https://github.com/szagoruyko/attention-transfer
    https://www.zhihu.com/question/36591394  Attention based model 是什么，它解决了什么问题？
- deformable convolution
    https://www.zhihu.com/question/57493889/answer/165287530  如何评价 MSRA 最新的 Deformable Convolutional Networks？
    https://github.com/oeway/pytorch-deform-conv
-   Spatial Transformer Networks
        http://blog.csdn.net/shaoxiaohu1/article/details/51809605
        https://www.zhihu.com/question/30817011 卷积神经网络提取图像特征时具有旋转不变性吗？
- 读相关文献


- 参考并复现 mxnet resnet 152的套路: https://github.com/YanWang2014/iNaturalist
    先用 resnet18和resnet50  # places365: AlexNet, ResNet18, ResNet50, DenseNet161
    Resnet 152: https://github.com/clcarwin/convert_torch_to_pytorch
- places: densenet 161, resnet 152及其他的imagenet模型inception-resnet v2等等的切换FLAG
    /home/wayne/python/kaggle/Ai_challenger/classification/ai_challenger_scene_train_20170904/upload/slim/new_test_visual
- ######### 各种冠军套路逐一实现  
    https://zhuanlan.zhihu.com/p/28084438?utm_source=itdadao&utm_medium=referral
    http://benanne.github.io/2015/03/17/plankton.html   
        gradually increased the intensity of the augmentation as our models started to overfit more
        Additional image features
        stochastic gradient descent (SGD) with Nesterov momentum
        Test-time augmentation
    海康威视等的场景比赛方案，场景分类相关论文
    ResNet相关
        https://github.com/facebook/fb.resnet.torch
        https://github.com/felixlaumon/kaggle-right-whale 第二名



后续2
- 各种集成(基础模型见后)
    https://github.com/YanWang2014/TensorFlow-Tutorials-1        https://www.youtube.com/watch?v=AVKZrPCW91A
    https://github.com/YanWang2014/PyTorchText  知乎看山杯 第一名 init 队解决方案
    https://zhuanlan.zhihu.com/p/28084438?utm_source=itdadao&utm_medium=referral
- ######### 掌握一下模型的架构思路/时空复杂度等理论理解方面（复现一些论文的技术）
    初始化，可视化，预处理，网络结构，Learning Rate，LRN，多任务，skip-connecttion, deformable convolution，attention。
    (理解是靠理论和直觉，设计是靠逻辑，靠循序渐进)
- 代码模块化: https://zhuanlan.zhihu.com/p/29024978
'''



'''
############ 相关的PyTorch预训练模型(最后可能用的基础模型有inception-resnet v2, densenet 161, resnet50, resnet 152，并混合imagenet和places 365)

# https://github.com/CSAILVision/places365
PyTorch Places365 models: AlexNet, ResNet18, ResNet50, DenseNet161

AlexNet-places365: deploy weights
GoogLeNet-places365: deploy weights
VGG16-places365: deploy weights
VGG16-hybrid1365: deploy weights
ResNet152-places365 fine-tuned from ResNet152-ImageNet: deploy weights
ResNet152-hybrid1365: deploy weights

ResNet152-places365 trained from scratch using Torch: torch model converted caffemodel:deploy weights. It is the original ResNet with 152 layers. On the validation set, the top1 error is 45.26% and the top5 error is 15.02%.
ResNet50-places365 trained from scratch using Torch: torch model. It is Preact ResNet with 50 layers. The top1 error is 44.82% and the top5 error is 14.71%.


# PyTorch imagenet models: http://pytorch.org/docs/master/torchvision/models.html
AlexNet
VGG
ResNet
SqueezeNet
DenseNet
Inception v3

# Pretrained ConvNets for pytorch: ResNeXt101, ResNet152, InceptionV4, InceptionResnetV2, etc.
https://github.com/Cadene/pretrained-models.pytorch
'''

#pkill -9 python
#nvidia-smi
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import json


'''
Important config
注意adam的结果可能会不是最好的
'''
arch = 'resnet18_places365'  
model_weight = 'whole_%s.pth.tar' % arch
use_gpu = 1
batch_size = 64
lr = 0.0001
num_epochs = 2

#Transforms on PIL.Image: RandomCrop, Pad
#RandomCrop: Crop the given PIL.Image at a random location.
#https://github.com/pytorch/vision/blob/master/torchvision/transforms.py
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224), #A crop of random size of (0.08 to 1.0) of the original size and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
        transforms.RandomHorizontalFlip(), #Horizontally flip the given PIL.Image randomly with a probability of 0.5.
        transforms.ToTensor(), #Converts a PIL.Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #channel = (channel - mean) / std
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),  #def resize(img, size, interpolation=Image.BILINEAR):
        transforms.CenterCrop(224), #Crops the given PIL.Image at the center.
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

if use_gpu == 1:
    model_conv = torch.load(model_weight, pickle_module=pickle)
else:
    model_conv = torch.load(model_weight, map_location=lambda storage, loc: storage, pickle_module=pickle) # model trained in GPU could be deployed in CPU machine like this!

#model_conv = torchvision.models.resnet18(pretrained=True)
#for param in model_conv.parameters():
#    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 80)

if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

#optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=lr, momentum=0.9)
#weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) 

# Decay LR by a factor of 0.1 every epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=200, gamma=0.1)







'''
load and transform data
'''
with open('../ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json', 'r') as f: #label文件
    label_raw_train = json.load(f)
with open('../ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json', 'r') as f: #label文件
    label_raw_val = json.load(f)


class SceneDataset(Dataset):

    def __init__(self, json_labels, root_dir, transform=None):
        """
        Args:
            json_labesl (list):read from official json file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_raw = json_labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_raw)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.label_raw[idx]['image_id'])
        image = Image.open(img_name)
        label = int(self.label_raw[idx]['label_id'])

        if self.transform:
            image = self.transform(image)

        return image, label


transformed_dataset_train = SceneDataset(json_labels=label_raw_train,
                                    root_dir='../ai_challenger_scene_train_20170904/scene_train_images_20170904',
                                           transform=data_transforms['train']
                                           )      
transformed_dataset_val = SceneDataset(json_labels=label_raw_val,
                                    root_dir='../ai_challenger_scene_validation_20170908/scene_validation_images_20170908',
                                           transform=data_transforms['val']
                                           )         
dataloader = {'train':DataLoader(transformed_dataset_train, batch_size=batch_size,shuffle=True, num_workers=8),
             'val':DataLoader(transformed_dataset_val, batch_size=batch_size,shuffle=False, num_workers=8)
             }
dataset_sizes = {'train': len(label_raw_train), 'val':len(label_raw_val)}
#use_gpu = torch.cuda.is_available()
#use_gpu = False

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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_model (model, criterion, optimizer, scheduler, num_epochs, total_steps):
    since = time.time()

    print('total_steps is %d' % total_steps)
    mystep = 0

    best_model_wts = model.state_dict()
    best_acc = 0.0
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if (epoch%10 == 0):
            torch.save(best_model_wts, ('%s_model_wts_%d.pth')% (arch, epoch))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            top1 = AverageMeter()
            top3 = AverageMeter()

            # Iterate over data.
            for data in dataloader[phase]:
                # get the inputs
                mystep = mystep + 1
                if(mystep%100 ==0):
                    duration = time.time() - since
                    print('step %d vs %d in %.0f s' % (mystep, total_steps, duration))

                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
#                print(type(labels)) # <class 'torch.autograd.variable.Variable'>
#                print(type(labels.data)) # <class 'torch.cuda.LongTensor'>
                prec1, prec3 = accuracy(outputs.data, labels.data, topk=(1, 3))
                top1.update(prec1[0], inputs.data.size(0))
                top3.update(prec3[0], inputs.data.size(0))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            #没测试batch_size不能被dataset_size整除时会不会有问题
            print('{} Loss: {:.6f} Acc: {:.6f}'.format(
                phase, epoch_loss, epoch_acc))
            print(' * Prec@1 {top1.avg:.6f} Prec@3 {top3.avg:.6f}'.format(top1=top1, top3=top3))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

        #if (epoch%10 == 0):
           # torch.save(best_model_wts, ('models/best_model_wts_%d.pth')% epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.6f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Train and evaluate

total_steps = 1.0 * num_epochs * (len(label_raw_train) + len(label_raw_val)) / batch_size
print(total_steps)
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs, total_steps)
torch.save(model_conv.state_dict(), ('%s_best_model_wts_final.pth')%arch)

