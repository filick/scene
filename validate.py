import os
import torch.backends.cudnn as cudnn
import torch.cuda
from common import AverageMeter, accuracy


def validate(scheme, use_gpu=torch.cuda.is_available()):
    model = scheme.model
    name = scheme.name
    loader = scheme.loader
    handler = scheme.handler

    best_check = 'checkpoint/' + name + '_best.pth.tar'

    if os.path.isfile(best_check):
        print("=> loading checkpoint '{}'".format(best_check))
        checkpoint = torch.load(best_check)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{0}".format(best_check))
    else:
        print("=> no checkpoint found at '{}'".format(best_check))
        return

    if use_gpu:
        cudnn.benchmark = True
        model = model.cuda()

    top1 = AverageMeter()
    top3 = AverageMeter()
    
    model.eval()

    for i, (inp, target) in enumerate(loader):

        if use_gpu:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(inp, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = handler(model, input_var)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        top1.update(prec1[0], inp.size(0))
        top3.update(prec3[0], inp.size(0))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
        .format(top1=top1, top3=top3))


if __name__ == '__main__':
    from basic import BasicTrainScheme, MultiScaleTrainScheme
    from scheme import ValidateScheme

    validate(ValidateScheme(MultiScaleTrainScheme()))