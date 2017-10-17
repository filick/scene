'''
http://blog.csdn.net/tfygg/article/details/70227388

torch.save()实现对网络结构和模型参数的保存。有两种保存方式：一是保存年整个神经网络的的结构信息和模型参数信息，save的对象是网络net；二是只保存神经网络的训练模型参数，save的对象是net.state_dict()。
torch.save(net1, '7-net.pth')                     # 保存整个神经网络的结构和模型参数    
torch.save(net1.state_dict(), '7-net_params.pth') # 只保存神经网络的模型参数    
对应上面两种保存方式，重载方式也有两种。对应第一种完整网络结构信息，重载的时候通过torch.load(‘.pth’)直接初始化新的神经网络对象即可。对应第二种只保存模型参数信息，需要首先导入对应的网络，通过net.load_state_dict(torch.load('.pth'))完成模型参数的重载。在网络比较大的时候，第一种方法会花费较多的时间。
# 保存和加载整个模型  
torch.save(model_object, 'model.pkl')  
model = torch.load('model.pkl')  
# 仅保存和加载模型参数(推荐使用)  
torch.save(model_object.state_dict(), 'params.pkl')  
model_object.load_state_dict(torch.load('params.pkl')) 
'''

'''
训练
Epoch 1/1
----------
step 1000 vs 1906 in 851 s
step 1100 vs 1906 in 937 s
step 1200 vs 1906 in 1023 s
step 1300 vs 1906 in 1109 s
step 1400 vs 1906 in 1194 s
step 1500 vs 1906 in 1281 s
step 1600 vs 1906 in 1368 s
step 1700 vs 1906 in 1454 s
train Loss: 0.059946 Acc: 0.232094
 * Prec@1 23.209414 Prec@3 40.462889
step 1800 vs 1906 in 1544 s
step 1900 vs 1906 in 1627 s
val Loss: 0.055907 Acc: 0.349157
 * Prec@1 34.915730 Prec@3 55.449438

Training complete in 27m 14s
Best val Acc: 0.349157

测试(val的准确率和训练时一样，但是loss和batch_size有关？)
batch_size = 32
test Loss: 0.292731 Acc: 0.003835
 * Prec@1 0.383523 Prec@3 0.710227
val Loss: 0.222152 Acc: 0.349157
 * Prec@1 34.915730 Prec@3 55.449438
batch_size = 64
test Loss: 0.073183 Acc: 0.003835
 * Prec@1 0.383523 Prec@3 0.710227
val Loss: 0.055887 Acc: 0.349157
 * Prec@1 34.915730 Prec@3 55.449438
batch_size =128
test Loss: 0.036591 Acc: 0.003835
 * Prec@1 0.383523 Prec@3 0.710227
val Loss: 0.027952 Acc: 0.349157
 * Prec@1 34.915730 Prec@3 55.449438
 
线下脚本验证
python scene_eval.py --submit ./submit_val.json
Evaluation time of your result: 4.208296 s
{'warning': [], 'score': '0.5544943820224719', 'error': []}

'''

'''
https://zhuanlan.zhihu.com/p/28084438?utm_source=itdadao&utm_medium=referral
3.11 测试时数据增强（TTA / Test Time Augmentation）

“上面我们提到训练时怎么使用数据增强，但是测试时数据增强（TTA）也可以对预测效果进行很大的提升。
具体做法也比较简单，我们可以将一个样本的八个方向都进行预测，获得八个预测概率，接着可以将八个概率直接平均，
也可以使用预测的类标签投票来获得最后结果。通过几轮测试，我们采取的是平均的方案，因为效果更好。”

#########################################################################
注意这里我们还不知道哪些数据增强比较重要，所以先以transdorms.five_crop和ten_crop为例, 实现一个框架
不采用softmax得到标签后再投票的办法，采用求概率向量平均值再进行预测的办法

https://github.com/pytorch/vision/blob/master/torchvision/transforms.py （目前没有旋转和各种仿射变换）
https://github.com/ncullen93/torchsample  (有旋转等仿射变换)

def five_crop(img, size):
    Crop the given PIL Image into four corners and the central crop.
    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
        tuple: tuple (tl, tr, bl, br, center) corresponding top left,
            top right, bottom left, bottom right and center crop.
            
class SpecialCrop(object):

    def __init__(self, size, crop_type=0):
        """
        Perform a special crop - one of the four corners or center crop
        Arguments
        ---------
        size : tuple or list
            dimensions of the crop
        crop_type : integer in {0,1,2,3,4}
            0 = center crop
            1 = top left crop
            2 = top right crop
            3 = bottom right crop
            4 = bottom left crop

class RandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.
        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p
        v : boolean
            whether to vertically flip w/ probability p
        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        
看来torchsample的变换和官方的不完全一样：
aug1
myhflip = [False]
mycrops = [0]
test Loss: 0.073180 Acc: 0.003551
 * Prec@1 0.355114 Prec@3 0.710227
val Loss: 0.055895 Acc: 0.348736
 * Prec@1 34.873596 Prec@3 55.491573

aug2
phases = ['val']
myhflip = [False]
mycrops = [0]
python scene_eval.py --submit ./submit2_val.json 
Evaluation time of your result: 3.088408 s
{'warning': [], 'error': [], 'score': '0.5549157303370786'}
与aug1一致

aug3
phases = ['val','test']
myhflip = [False, True]
mycrops = [0,1]
python scene_eval.py --submit ./submit3_val.json 
Evaluation time of your result: 3.039888 s
{'error': [], 'score': '0.5276685393258427', 'warning': []}
变差了？

'''
#pkill -9 python
#nvidia-smi
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import time
import json
import torchsample
import numpy as np
import copy
#import pickle
from sklearn.utils.extmath import softmax

'''
Important config
'''
arch = 'resnet18_places365'  # AlexNet, ResNet18, ResNet50, DenseNet161
model_conv = torchvision.models.resnet18()
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 80)
for param in model_conv.parameters():
    param.requires_grad = False #节省显存
model_weight = '%s_best_model_wts_final.pth' % arch

use_gpu = 1
batch_size = 64
mystep = 0    
phases = ['val','test']
myhflip = [False, True]
mycrops = [0,1,2,3,4]


'''
load pretrained model
'''

if use_gpu == 1:
    model_conv.load_state_dict(torch.load(model_weight)) 
else:
    model_conv.load_state_dict(torch.load(model_weight)) # = torch.load(model_weight, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!

if use_gpu:
    model_conv = model_conv.cuda()
#criterion = nn.CrossEntropyLoss()


'''
load data
'''
with open('../ai_challenger_scene_test_a_20170922/scene_test_annotations.json', 'r') as f: #label文件, 测试的是我自己生成的
    label_raw_test = json.load(f)
with open('../ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json', 'r') as f: #label文件
    label_raw_val = json.load(f)
dataset_sizes = {'test': len(label_raw_test), 'val':len(label_raw_val)}


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
        img_name_raw = self.label_raw[idx]['image_id']
        image = Image.open(img_name)
        label = int(self.label_raw[idx]['label_id'])

        if self.transform:
            image = self.transform(image)

        return image, label, img_name_raw

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
    """Computes the precision@k for the specified values of k
    output: logits
    target: labels
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
        
#    print(output)
#    print(target)
#    print(pred)
#    print(correct)
#    print(res)
    
#    print(type(pred))
    pred_list = pred.tolist()  #[[14, 13], [72, 15], [74, 11]]
#    print(pred_list)
    return res, pred_list

def batch_to_list_of_dicts(indices, image_ids):  #indices2 是预测的labels
    '''
    pred_list = pred.tolist()  #[[14, 13], [72, 15], [74, 11]]
    print(img_name_raw) #('ed531a55d4887dc287119c3f6ebf7eb162bed6cf.jpg', '520036616eb2594b6e9d41b0415deea607e8de12.jpg')
    '''
    result = [] #[{"image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg","label_id": [5,18,32]}]
    dict_ = {}
    for item in range(len(image_ids)):
        dict_ ['image_id'] = image_ids[item]
        dict_['label_id'] = [indices[0][item], indices[1][item], indices[2][item]]
        result.append(dict_)
        dict_ = {}
    return result

def batch_to_list_of_dicts2(logits, image_ids):
    result = [] 
    dict_ = {}
    for item in range(len(image_ids)):
        dict_ ['image_id'] = image_ids[item]
        dict_['logits'] = logits[item,:]
        result.append(dict_)
        dict_ = {}
    return result

my_aug_logits = {}
my_aug_logits2 = {}

def test_model (model, myid, total_steps, mystep):
    since = time.time()

    for phase in phases:
        
        model.train(False)  # Set model to evaluate mode

#        running_loss = 0.0
#        running_corrects = 0
#        top1 = AverageMeter()
#        top3 = AverageMeter()
#        results = []
        
        aug_logits = {}

        for data in dataloader[phase]:
            mystep = mystep + 1
            if(mystep%10 ==0):
                duration = time.time() - since
                print('step %d vs %d in %.0f s' % (mystep, total_steps, duration))

            inputs, labels, img_name_raw= data
            #print(img_name_raw) #('ed531a55d4887dc287119c3f6ebf7eb162bed6cf.jpg', '520036616eb2594b6e9d41b0415deea607e8de12.jpg')

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = model(inputs) # torch.FloatTensor of size batch_sizex80
            #outputs2 = torch.nn.Softmax(outputs)
#            print(outputs)
#            print(outputs.data)
#            aug_logits += batch_to_list_of_dicts2(outputs.data, img_name_raw)
#            aug_logits = np.vstack((aug_logits,outputs.numpy()))
            ee = outputs.cpu().data.numpy()
            if mystep == 1:
                print(ee[0,:])
                print(img_name_raw[0])
            for item in range(len(img_name_raw)):
#                print(ee[item,:])
#                print(softmax([ee[item,:]])[0])
                aug_logits[img_name_raw[item]] = softmax([ee[item,:]])[0] #防止多线程啥的改变了图片顺序，还是按照id保存比较保险
                #print(aug_logits[img_name_raw[item]])
            
#            _, preds = torch.max(outputs.data, 1)
#            loss = criterion(outputs, labels)
#
#            running_loss += loss.data[0]
#            running_corrects += torch.sum(preds == labels.data)
#
#            res, pred_list = accuracy(outputs.data, labels.data, topk=(1, 3))
#            prec1 = res[0]
#            prec3 = res[1]
#            top1.update(prec1[0], inputs.data.size(0))
#            top3.update(prec3[0], inputs.data.size(0))
#
#            results += batch_to_list_of_dicts(pred_list, img_name_raw)
#
#        epoch_loss = running_loss / dataset_sizes[phase]
#        epoch_acc = running_corrects / dataset_sizes[phase]
#
#        print('{} Loss: {:.6f} Acc: {:.6f}'.format(
#            phase, epoch_loss, epoch_acc))
#        print(' * Prec@1 {top1.avg:.6f} Prec@3 {top3.avg:.6f}'.format(top1=top1, top3=top3))
#        
#        with open(('logits_%s_%s.json'%(phase, str(myid))), 'w') as f:
#            json.dump(aug_logits, f)
        my_aug_logits2[phase] = aug_logits
    my_aug_logits[str(myid)] = my_aug_logits2

    return 0



######################################################################
# val and test

myid = 0
for hflip in myhflip:
    for crops in mycrops:
        
        data_transforms = {
            'test': transforms.Compose([
                transforms.Scale(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                torchsample.transforms.RandomFlip(hflip,False,2),# p>1无所谓，就是确保水平翻转
                torchsample.transforms.SpecialCrop([224,224], crops),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Scale(256),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                torchsample.transforms.RandomFlip(hflip,False,2),# p>1无所谓，就是确保水平翻转
                torchsample.transforms.SpecialCrop([224,224], crops),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        
        transformed_dataset_test = SceneDataset(json_labels=label_raw_test,
                                            root_dir='../ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922',
                                                   transform=data_transforms['test']
                                                   )      
        transformed_dataset_val = SceneDataset(json_labels=label_raw_val,
                                            root_dir='../ai_challenger_scene_validation_20170908/scene_validation_images_20170908',
                                                   transform=data_transforms['val']
                                                   )         
        dataloader = {'test':DataLoader(transformed_dataset_test, batch_size=batch_size,shuffle=False, num_workers=8),
                     'val':DataLoader(transformed_dataset_val, batch_size=batch_size,shuffle=False, num_workers=8)
                     }
        total_steps = 1.0  * (len(label_raw_test) + len(label_raw_val)) / batch_size *(len(mycrops) * len(myhflip))
        print(total_steps)
        test_model(model_conv, myid, total_steps, mystep)  #改了，将每种TTA的预测概率向量保存进字典
        myid += 1

'''
合并不同TTA的结果
'''

#for phase in ['test', 'val']:
#    final_logits = my_aug_logits['0'][phase]# [{"image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg","logits": [5,18,32]}]
#    if phase=='test'
#        for img in label_raw_test:
#            img['image_id']
#    for item in range(myid):
#        my_aug_logits[str(myid)]

def np_to_list(array):
    if(len(array)!=80):
        print(array)
    return ((-array).argsort()[:3]).astype('int32').tolist()
    
for phase in phases:
    final_results = []
    
    '''小心，字典的深拷贝'''
    final_logits = copy.deepcopy(my_aug_logits['0'][phase])  # {'ed531a55d4887dc287119c3f6ebf7eb162bed6cf.jpg': [1,2,3,..80]}
    #print(final_logits['0c58107693263d32551209512d858246e925fe29.jpg'])
    #print(sum(final_logits['0c58107693263d32551209512d858246e925fe29.jpg']))
    for key in final_logits:
        final_logits[key] *= 0
    #print(final_logits)
    #print(my_aug_logits['0'][phase])
    if phase=='test':
        image_ids = label_raw_test #[{"image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg","label_id": 5}]
    else:
        image_ids = label_raw_val
    
    for item in image_ids:
        image_id = item['image_id']
        for it in range(myid):
            final_logits[image_id] += my_aug_logits[str(it)][phase][image_id]
    
    #print(np_to_list(final_logits['0c58107693263d32551209512d858246e925fe29.jpg']))
    results = [] #[{"image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg","label_id": [5,18,32]}]
    dict_ = {}
    for item in image_ids:
        dict_ ['image_id'] = item['image_id']
        #print(final_logits[item['image_id']])
        dict_['label_id'] = np_to_list(final_logits[item['image_id']])
        results.append(dict_)
        dict_ = {}
    #print(results[:20])
    #print(type(results))

#    with open('outfile', 'wb') as fp:
#        pickle.dump(results, fp)
#    with open ('outfile', 'rb') as fp:
#        itemlist = pickle.load(fp)
#        
    with open(('submit3_%s.json'%phase), 'w') as f:
        json.dump(results, f)
        
'''
    raise TypeError(repr(o) + " is not JSON serializable")

TypeError: 40 is not JSON serializable
'''
