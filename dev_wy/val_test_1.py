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


data_transforms = {
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = 1
batch_size = 64#32








'''
load pretrained model
'''

if use_gpu == 1:
    model_conv.load_state_dict(torch.load(model_weight)) 
else:
    model_conv.load_state_dict(torch.load(model_weight)) # = torch.load(model_weight, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!


'''
load and transform data
'''
with open('../ai_challenger_scene_test_a_20170922/scene_test_annotations.json', 'r') as f: #label文件, 测试的是我自己生成的
    label_raw_test = json.load(f)
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
        img_name_raw = self.label_raw[idx]['image_id']
        image = Image.open(img_name)
        label = int(self.label_raw[idx]['label_id'])

        if self.transform:
            image = self.transform(image)

        return image, label, img_name_raw


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
dataset_sizes = {'test': len(label_raw_test), 'val':len(label_raw_val)}
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

def test_model (model, criterion):
    since = time.time()

    mystep = 0    

    for phase in ['test', 'val']:
        
        model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        top1 = AverageMeter()
        top3 = AverageMeter()
        results = []

        # Iterate over data.
        for data in dataloader[phase]:
            # get the inputs
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
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
#            print(loss)
#            print(loss.data)
#            print(loss.data[0])

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)
#                print(type(labels)) # <class 'torch.autograd.variable.Variable'>
#                print(type(labels.data)) # <class 'torch.cuda.LongTensor'>
#            print(outputs)
#            print(outputs.data)
#            print(labels)
#            print(labels.data)
            res, pred_list = accuracy(outputs.data, labels.data, topk=(1, 3))
            prec1 = res[0]
            prec3 = res[1]
            top1.update(prec1[0], inputs.data.size(0))
            top3.update(prec3[0], inputs.data.size(0))
#            print(prec1)
#            print(prec3)
#            print(img_name_raw)
#            print(type(img_name_raw))
            results += batch_to_list_of_dicts(pred_list, img_name_raw)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        #没测试batch_size不能被dataset_size整除时会不会有问题
        print('{} Loss: {:.6f} Acc: {:.6f}'.format(
            phase, epoch_loss, epoch_acc))
        print(' * Prec@1 {top1.avg:.6f} Prec@3 {top3.avg:.6f}'.format(top1=top1, top3=top3))
        
        with open(('submit_%s.json'%phase), 'w') as f:
            json.dump(results, f)

    return 0


if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()


######################################################################
# val and test
total_steps = 1.0  * (len(label_raw_test) + len(label_raw_val)) / batch_size
print(total_steps)
test_model(model_conv, criterion)
