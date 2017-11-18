from torch.optim import Optimizer, required
from torch.autograd import Variable


class AutoLRSGD(Optimizer):

    def __init__(self, params, max_lr=required, ideal_loss=required,
                 strength=required, weight_decay=0):
        defaults = dict(max_lr=max_lr, ideal_loss=ideal_loss, 
                        strength=strength, weight_decay=weight_decay)
        super(AutoLRSGD, self).__init__(params, defaults)

    def step(self, closure):
        loss = closure()
        if isinstance(loss, Variable):
            loss = loss.data

        for group in self.param_groups:
            max_lr = group['max_lr']
            ideal_loss = group['ideal_loss']
            strength = group['strength']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                lr = loss.add(-ideal_loss).div(d_p).mul(strength).clamp(max=max_lr)
                p.data.add_(lr.neg(), d_p)

        return loss
