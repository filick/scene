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


arch = 'resnet18'
pretrained = 'places'
evaluate = False
checkpoint_filename = arch + '_' + pretrained
try_resume = False
print_freq = 10
start_epoch = 0
use_gpu = torch.cuda.is_available()
class_aware = True

# training parameters:
BATCH_SIZE = 128
INPUT_WORKERS = 8
epochs = 90
lr = 0.0001
betas=(0.9, 0.999)
eps=1e-08
weight_decay=0


latest_check = 'checkpoint/' + checkpoint_filename + '_latest.pth.tar'
best_check = 'checkpoint/' + checkpoint_filename + '_best.pth.tar'


def run():
    model = load_model(arch, pretrained, use_gpu=use_gpu)

    if use_gpu:
        if arch.lower().startswith('alexnet') or arch.lower().startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        best_prec1 = 0

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
        train_set = data.ChallengerSceneFolder(data.TRAIN_ROOT, data_transforms('train'))
        train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=BATCH_SIZE, shuffle=False,
                sampler=ClassAwareSampler.ClassAwareSampler(train_set),
                num_workers=INPUT_WORKERS, pin_memory=use_gpu)
    else:
        train_loader = torch.utils.data.DataLoader(
                data.ChallengerSceneFolder(data.TRAIN_ROOT, data_transforms('train')),
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=INPUT_WORKERS, pin_memory=use_gpu)
        
    val_loader = torch.utils.data.DataLoader(
            data.ChallengerSceneFolder(data.VALIDATION_ROOT, data_transforms('validation')),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=INPUT_WORKERS, pin_memory=use_gpu)


    criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) 


    if evaluate:
        validate(val_loader, model, criterion)

    else:

        for epoch in range(start_epoch, epochs):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


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
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    return top3.avg


def validate(val_loader, model, criterion):
    return _each_epoch('validate', val_loader, model, criterion)


def train(train_loader, model, criterion, optimizer, epoch):
    return _each_epoch('train', train_loader, model, criterion, optimizer, epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 5 every 10 epochs"""
    lr_new = lr * (0.2 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new


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