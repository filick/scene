import os
import time
import shutil
import torch.backends.cudnn as cudnn
import torch.cuda
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from hyperboard import Agent
from common import AverageMeter, accuracy
from opt import AutoLRSGD


def train(scheme, epochs, agent=None,
          try_resume=True, print_freq=10, 
          use_gpu=torch.cuda.is_available()):
    name = scheme.name
    model = scheme.model
    train_loader, validate_loader = scheme.train_loader, scheme.validate_loader
    criterion = scheme.criterion
    optimizer, lr_schedule = scheme.optimizer, scheme.lrschedule
    hyperparameters = scheme.hyperparams

    latest_check = 'checkpoint/' + name + '_latest.pth.tar'
    best_check = 'checkpoint/' + name + '_best.pth.tar'

    if agent is not None:
        names = {}
        labels = ['train_loss', 'train_acc1', 'train_acc3']
        if validate_loader is not None:
            labels += ['validate_loss', 'validate_acc1', 'validate_acc3']
        for label in labels:
            metric = label.split("_")[1]
            if metric[-1] == '1' or metric[-1] == '3':
                metric = metric[0:-1]
            hyperparameters['label'] = label
            names[label] = agent.register(hyperparameters, metric)
        hyperparameters['label'] = 'lr'
        names['lr'] = agent.register(hyperparameters, 'lr')


    best_prec3 = 0
    start_epoch = 0

    if try_resume:
        if os.path.isfile(latest_check):
            print("=> loading checkpoint '{}'".format(latest_check))
            checkpoint = torch.load(latest_check)
            start_epoch = checkpoint['epoch']
            best_prec3 = checkpoint['best_prec3']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(latest_check, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(latest_check))

    if use_gpu:
        cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()


    def _each_epoch(mode, loader, epoch):
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
        for i, (inp, target) in enumerate(loader):
            data_time.update(time.time() - end)

            if use_gpu:
                target = target.cuda(async=True)
            input_var = torch.autograd.Variable(inp, volatile=(mode != 'train'))
            target_var = torch.autograd.Variable(target, volatile=(mode != 'train'))

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.data[0], inp.size(0))
            top1.update(prec1[0], inp.size(0))
            top3.update(prec3[0], inp.size(0))

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                if isinstance(optimizer, AutoLRSGD):
                    def closure():
                        return loss.data
                    optimizer.step(closure)
                else:
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
                    if agent is not None:
                        index = i + epoch * len(loader)
                        agent.append(names['train_loss'], index, losses.val)
                        agent.append(names['train_acc1'], index, top1.val)
                        agent.append(names['train_acc3'], index, top3.val)

        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
            .format(top1=top1, top3=top3))

        if mode == 'validate':
            index = (epoch + 1) * len(train_loader) - 1
            agent.append(names['validate_loss'], index, losses.avg)
            agent.append(names['validate_acc1'], index, top1.avg)
            agent.append(names['validate_acc3'], index, top3.avg)

        return losses, top1, top3


    def validate_epoch(val_loader, epoch):
        return _each_epoch('validate', val_loader, epoch)


    def train_epoch(train_loader, epoch):
        return _each_epoch('train', train_loader, epoch)


    def save_checkpoint(state, is_best):
        torch.save(state, latest_check)
        if is_best:
            shutil.copyfile(latest_check, best_check)


    for epoch in range(start_epoch, epochs):
        if isinstance(lr_schedule, _LRScheduler):
            lr_schedule.step()

        agent.append(names['lr'], epoch * len(train_loader), optimizer.param_groups[0]['lr'])
        # train for one epoch
        loss, top1, top3 = train_epoch(train_loader, epoch)
        prec3 = top3.avg

        # evaluate on validation set
        if validate_loader is not None:
            loss, top1, top3 = validate_epoch(validate_loader, epoch)
            prec3 = top3.avg

        if isinstance(lr_schedule, ReduceLROnPlateau):
            lr_schedule.step(loss.avg)

        # remember best prec@1 and save checkpoint
        is_best = prec3 > best_prec3
        best_prec3 = max(prec3, best_prec3)
        save_checkpoint({
            'epoch': epoch + 1,
            'hyperparams': hyperparameters,
            'state_dict': model.state_dict(),
            'best_prec3': best_prec3,
        }, is_best)


if __name__ == '__main__':
    from basic import BasicTrainScheme, MultiScaleTrainScheme
    trainscheme = BasicTrainScheme()
    agent = Agent()
    train(trainscheme, epochs=100, agent=agent,
          try_resume=True, print_freq=20, 
          use_gpu=torch.cuda.is_available())