'''
torch转过来的模型怎么控制每一层的lr？
spp layer
'''

import data
from model import load_model
import torch.utils.data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import shutil
import os
from utils import ClassAwareSampler
from config import data_transforms
from hyperboard import Agent

arch = 'resnet152' # preact_resnet50, resnet152
pretrained = 'places' #imagenet
evaluate = False
checkpoint_filename = arch + '_' + pretrained
try_resume = False
print_freq = 10
start_epoch = 0
use_gpu = torch.cuda.is_available()
class_aware = True
AdaptiveAvgPool=True
SPP = False
num_levels = 3
pool_type = 'avg_pool'
input_size = 256 #[224, 256, 384, 480, 640] 
train_scale = 256
test_scale = 256
train_transform = 'train2'
lr_decay = 0.5

# training parameters:
BATCH_SIZE = 130
INPUT_WORKERS = 8
epochs = 100
lr = 0.001 

if_fc = True #是否先训练最后新加的层
lr1 = 0.000 #if_fc = True, 里面的层先不动
lr2 = 0.1 #if_fc = True, 先学好最后一层
lr2_min = 0.0009 #lr2每次除以10降到lr2_min，然后lr2 = lr, lr1 = lr2/slow
slow = 1 #if_fc = True, lr1比lr2慢的倍数

weight_decay=0 #.05 #0.0005 #0.0001  0.05太大。试下0.01?
optim_type = 'SGD' #Adam SGD
betas=(0.9, 0.999)
eps=1e-08 # 0.1的话一开始都是prec3 4.几
momentum = 0.9


hyperparameters = {
    'arch': arch,
    'pretrained': pretrained,
    'SPP': SPP,
    'num_levels': num_levels,
    'pool_type': pool_type,
    'class_aware': class_aware,
    'batch_size': BATCH_SIZE,
    'epochs': epochs,
    'if_fc': if_fc,
    'lr': lr,
    'lr1': lr1,
    'lr2': lr2,
    'lr2_min': lr2_min,
    'slow': slow,
    'optim_type': optim_type,
    'weight_decay': weight_decay,
    'eps': eps,
    'input_size': input_size,
    'train_scale': train_scale,
    'test_scale': test_scale,
    'train_transform': train_transform,
    'lr_decay': lr_decay,
    'monitoring': None
}

monitoring = ['train_loss', 'train_accu1', 'train_accu3', 'valid_loss', 'valid_accu1', 'valid_accu3']
names = {}

agent = Agent()
for m in monitoring:
    hyperparameters['result'] = m
    metric = m.split('_')[-1]
    name = agent.register(hyperparameters, metric)
    names[m] = name

latest_check = 'checkpoint/' + checkpoint_filename + '_latest.pth.tar'
best_check = 'checkpoint/' + checkpoint_filename + '_best.pth.tar'


def run():
    model = load_model(arch, pretrained, use_gpu=use_gpu, AdaptiveAvgPool=AdaptiveAvgPool, SPP=SPP, num_levels=num_levels, pool_type=pool_type)
    if if_fc:
        if pretrained == 'imagenet' or arch == 'resnet50' or arch == 'resnet18':
            ignored_params = list(map(id, model.fc.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params,
                                 model.parameters())
            lr_dicts = [{'params': base_params, 'lr':lr1}, 
                         {'params': model.fc.parameters(), 'lr':lr2}]
            
        elif pretrained =='places':
            if arch == 'preact_resnet50':
                lr_dicts = list()
                lr_dicts.append({'params': model._modules['12']._modules['1'].parameters(), 'lr':lr2})
                for _, index in enumerate(model._modules):
                    if index != '12':
                        lr_dicts.append({'params': model._modules[index].parameters(), 'lr':lr1})
                    else:
                        for index2,_ in enumerate(model._modules[index]):
                            if index2 !=1:
                                lr_dicts.append({'params': model._modules[index]._modules[str(index2)].parameters(), 'lr':lr1})
            elif arch == 'resnet152':
                lr_dicts = list()
                lr_dicts.append({'params': model._modules['10']._modules['1'].parameters(), 'lr':lr2})
                for _, index in enumerate(model._modules):
                    if index != '10':
                        lr_dicts.append({'params': model._modules[index].parameters(), 'lr':lr1})
                    else:
                        for index2,_ in enumerate(model._modules[index]):
                            if index2 !=1:
                                lr_dicts.append({'params': model._modules[index]._modules[str(index2)].parameters(), 'lr':lr1})
                                
    if use_gpu:
        if arch.lower().startswith('alexnet') or arch.lower().startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        best_prec1 = 0
        best_loss1 = 10000

    if try_resume:
        if os.path.isfile(latest_check):
            print("=> loading checkpoint '{}'".format(latest_check))
            checkpoint = torch.load(latest_check)
            global start_epoch 
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(latest_check, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(latest_check))

    cudnn.benchmark = True


    if class_aware:
        train_set = data.ChallengerSceneFolder(data.TRAIN_ROOT, data_transforms(train_transform,input_size, train_scale, test_scale))
        train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=BATCH_SIZE, shuffle=False,
                sampler=ClassAwareSampler.ClassAwareSampler(train_set),
                num_workers=INPUT_WORKERS, pin_memory=use_gpu)
    else:
        train_loader = torch.utils.data.DataLoader(
                data.ChallengerSceneFolder(data.TRAIN_ROOT, data_transforms(train_transform,input_size, train_scale, test_scale)),
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=INPUT_WORKERS, pin_memory=use_gpu)
        
    val_loader = torch.utils.data.DataLoader(
            data.ChallengerSceneFolder(data.VALIDATION_ROOT, data_transforms('validation',input_size, train_scale, test_scale)),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=INPUT_WORKERS, pin_memory=use_gpu)


    criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    

    if if_fc:
        if optim_type == 'Adam':
                optimizer = optim.Adam(lr_dicts,
                                         betas=betas, eps=eps, weight_decay=weight_decay) 
        elif optim_type == 'SGD':
                optimizer = optim.SGD(lr_dicts, 
                                         momentum=momentum, weight_decay=weight_decay)
    else:
        if optim_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) 
        elif optim_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    if evaluate:
        validate(val_loader, model, criterion)

    else:

        for epoch in range(start_epoch, epochs):

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1, loss1= validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
          
            
            is_best_loss = (loss1 <= best_loss1)
            if is_best_loss:
                best_loss1 = loss1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best)
            else:
                my_check = torch.load(best_check)
                model.load_state_dict(my_check['state_dict'])
                adjust_learning_rate(optimizer, epoch, if_fc)


def _each_epoch(mode, loader, model, criterion, optimizer=None, epoch=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    if mode == 'train':
        model.train()
    else:
        model.eval()

    end = time.time()
    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - end)

        if use_gpu:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=(mode != 'train'))
        target_var = torch.autograd.Variable(target, volatile=(mode != 'train'))

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if mode == 'train':
#            if i % print_freq == 0:
#                print('Epoch: [{0}][{1}/{2}]\t'
#                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                    'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
#                    epoch, i, len(loader), batch_time=batch_time,
#                    data_time=data_time, loss=losses, top1=top1, top3=top3))
        index = epoch
        agent.append(names['train_loss'], index, losses.avg)
        agent.append(names['train_accu1'], index, top1.avg)
        agent.append(names['train_accu3'], index, top3.avg)
    elif mode == 'validate':
        index = epoch
        agent.append(names['valid_loss'], index, losses.avg)
        agent.append(names['valid_accu1'], index, top1.avg)
        agent.append(names['valid_accu3'], index, top3.avg)

    print(' *Epoch:[{0}] Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f} Loss {loss.avg:.4f}'
          .format(epoch,top1=top1, top3=top3, loss=losses))

    return top3.avg, losses.avg


def validate(val_loader, model, criterion, epoch):
    return _each_epoch('validate', val_loader, model, criterion, optimizer=None, epoch=epoch)


def train(train_loader, model, criterion, optimizer, epoch):
    return _each_epoch('train', train_loader, model, criterion, optimizer, epoch)


def adjust_learning_rate(optimizer, epoch, if_fc):
    #lr_new = lr * (lr_decay1 ** (epoch // lr_decay2))
    global lr

    if if_fc:
        global lr1
        global lr2 #最后一层
        if lr2 >= lr2_min:
            lr2 = lr2 * 0.1
        else:
            lr2 = lr
            lr1 = lr2/slow
            lr = lr * lr_decay
        print('lr2, lr1, lr')
        print(lr2)
        print(lr1)
        print(lr)
        param_groups = optimizer.param_groups
        for param_group in param_groups:
            param_group['lr'] = lr1
        param_groups[0]['lr'] = lr2
    else:
        lr = lr * lr_decay
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best):
    torch.save(state, latest_check)
    if is_best:
        shutil.copyfile(latest_check, best_check)


if __name__ == '__main__':
    run()
