import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
from model import load_model
from config import my_transform_multiscale_test
import copy
import pickle

'''
简单配置的参考结果:
    
phases = ['val']
test_scales = [224]
flip_flags = [False]
55.421348

test_scales = [256]
55.800562

test_scales = [384]
flip_flags = ['False']
54.185393
'''

arch = 'resnet152'
pretrained = 'places'
phases = ['test', 'val'] 
test_scales = [224, 256, 384, 480, 640]  #注意，与训练时用的尺度差别太大可能导致结果不好
flip_flags = [True, False]   

use_gpu = torch.cuda.is_available()
batch_size = 128 #64
INPUT_WORKERS = 8
checkpoint_filename = arch + '_' + pretrained
best_check = 'checkpoint/' + checkpoint_filename + '_best.pth.tar' #.tar



#load_model_fcn
model_conv = load_model(arch, pretrained, use_gpu=use_gpu, AdaptiveAvgPool=True)
for param in model_conv.parameters():
    param.requires_grad = False #节省显存
    
best_checkpoint = torch.load(best_check)
if use_gpu:
    if arch.lower().startswith('alexnet') or arch.lower().startswith('vgg'):
        model_conv.features = nn.DataParallel(model_conv.features)
        model_conv.cuda()
        model_conv.load_state_dict(best_checkpoint['state_dict']) 
    else:
        model_conv = nn.DataParallel(model_conv).cuda()
        model_conv.load_state_dict(best_checkpoint['state_dict']) 



with open('data/test/scene_test_annotations.json', 'r') as f: #label文件, 测试的是我自己生成的
    label_raw_test = json.load(f)
with open('data/validation/scene_validation_annotations_20170908.json', 'r') as f: #label文件
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

dataset_sizes = {'test': len(label_raw_test), 'val':len(label_raw_val)}


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
        

    pred_list = pred.tolist()  #[[14, 13], [72, 15], [74, 11]]
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

my_aug_softmax = {}
my_aug_softmax2 = {}
def test_model (model, criterion, myid):

    for phase in phases:
        
        model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        top1 = AverageMeter()
        top3 = AverageMeter()
        results = []
        aug_softmax = {}

        # Iterate over data.
        for data in dataloader[phase]:
            # get the inputs
            global mystep
            mystep = mystep + 1
#            if(mystep%10 ==0):
#                duration = time.time() - since
#                print('step %d vs %d in %.0f s' % (mystep, total_steps, duration))

            inputs, labels, img_name_raw= data
            #print(img_name_raw) #('ed531a55d4887dc287119c3f6ebf7eb162bed6cf.jpg', '520036616eb2594b6e9d41b0415deea607e8de12.jpg')

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            crop_logits = model(inputs) 
            crop_softmax = nn.functional.softmax(crop_logits)
            temp = crop_softmax.cpu().data.numpy()
            for item in range(len(img_name_raw)):
                aug_softmax[img_name_raw[item]] = temp[item,:] #防止多线程啥的改变了图片顺序，还是按照id保存比较保险
                
            _, preds = torch.max(crop_softmax.data, 1)
            loss = criterion(crop_logits, labels)

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

            res, pred_list = accuracy(crop_softmax.data, labels.data, topk=(1, 3))
            prec1 = res[0]
            prec3 = res[1]
            top1.update(prec1[0], inputs.data.size(0))
            top3.update(prec3[0], inputs.data.size(0))
            
            results += batch_to_list_of_dicts(pred_list, img_name_raw)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        print('{} Loss: {:.6f} Acc: {:.6f}'.format(
            phase, epoch_loss, epoch_acc))
        print(' * Prec@1 {top1.avg:.6f} Prec@3 {top3.avg:.6f}'.format(top1=top1, top3=top3))
        
        my_aug_softmax2[phase] = aug_softmax
    my_aug_softmax[str(myid)] = my_aug_softmax2
    return 0



criterion = nn.CrossEntropyLoss()


######################################################################
# val and test
total_steps = 1.0  * len(flip_flags) * len(test_scales) * (len(label_raw_test) + len(label_raw_val)) / batch_size
print(total_steps)
mystep = 0    

myid = 0
for flip_flag in flip_flags:
    for varied_scale in test_scales:
        print('flip = %s' % flip_flag)
        print('varied scale = %d' % varied_scale)
        transformed_dataset_test = SceneDataset(json_labels=label_raw_test,
                                    root_dir='/home/member/fuwang/projects/scene/data/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922',
                                           transform=my_transform_multiscale_test(varied_scale,flip_flag)
                                           )      
        transformed_dataset_val = SceneDataset(json_labels=label_raw_val,
                                    root_dir='/home/member/fuwang/projects/scene/data/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',
                                           transform=my_transform_multiscale_test(varied_scale,flip_flag)
                                           )           
        
        dataloader = {'test':DataLoader(transformed_dataset_test, batch_size=batch_size,shuffle=False, num_workers=INPUT_WORKERS),
                     'val':DataLoader(transformed_dataset_val, batch_size=batch_size,shuffle=False, num_workers=INPUT_WORKERS)
                     }
        test_model(model_conv, criterion, myid)
        myid += 1


'''
集成不同数据增强方案的test结果
'''

def np_to_list(array):
    if(len(array)!=80):
        print(array)
    return ((-array).argsort()[:3]).astype('int32').tolist()
    
for phase in phases:
    final_results = []
    
    '''小心，字典的深拷贝'''
    final_softmax = copy.deepcopy(my_aug_softmax['0'][phase])  # {'ed531a55d4887dc287119c3f6ebf7eb162bed6cf.jpg': [1,2,3,..80]}

    for key in final_softmax:
        final_softmax[key] *= 0

    if phase=='test':
        image_ids = label_raw_test #[{"image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg","label_id": 5}]
    else:
        image_ids = label_raw_val
    
    for item in image_ids:
        image_id = item['image_id']
        for it in range(myid):
            final_softmax[image_id] += my_aug_softmax[str(it)][phase][image_id]
        final_softmax[image_id] /= len(range(myid))    
        
    with open(('submit/%s_softmax4_%s.txt'%(checkpoint_filename, phase)), 'wb') as handle:
        pickle.dump(final_softmax, handle)
    
    results = [] #[{"image_id":"a0563eadd9ef79fcc137e1c60be29f2f3c9a65ea.jpg","label_id": [5,18,32]}]
    dict_ = {}
    for item in image_ids:
        dict_ ['image_id'] = item['image_id']
        #print(final_logits[item['image_id']])
        dict_['label_id'] = np_to_list(final_softmax[item['image_id']])
        results.append(dict_)
        dict_ = {}
 
    with open(('submit/%s_submit4_%s.json'%(checkpoint_filename, phase)), 'w') as f:
        json.dump(results, f)
