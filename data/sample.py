#https://github.com/pytorch/pytorch/pull/3062

import random
import torch
from torch.utils.data.sampler import Sampler


class RandomCycleIter:
    """Randomly iterate element in each cycle

    Example:
        >>> rand_cyc_iter = RandomCycleIter([1, 2, 3])
        >>> [next(rand_cyc_iter) for _ in range(10)]
        [2, 1, 3, 2, 3, 1, 1, 2, 3, 2]
    """
    def __init__(self, data):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1
        if self.i == self.length:
            self.i = 0
            random.shuffle(self.data_list)
        return self.data_list[self.i]

    next = __next__  # Py2


def class_aware_sample_generator(cls_iter, data_iter_list, n):
    i = 0
    while i < n:
        yield next(data_iter_list[next(cls_iter)])
        i += 1


class ClassAwareSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from

    Implemented Class-Aware Sampling: https://arxiv.org/abs/1512.05830
    Li Shen, Zhouchen Lin, Qingming Huang, Relay Backpropagation for Effective
    Learning of Deep Convolutional Neural Networks, ECCV 2016
    By default num_samples equals to number of samples in the largest class
    multiplied by num of classes such that all samples can be sampled.
    """

    def __init__(self, data_source, num_samples=0):
        self.data_source = data_source
        n_cls = len(data_source.classes)
        self.class_iter = RandomCycleIter(range(n_cls))
        cls_data_list = [list() for _ in range(n_cls)]
        for i, (_, label) in enumerate(data_source.imgs):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        if num_samples:
            self.num_samples = num_samples
        else:
            self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list, self.num_samples)

    def __len__(self):
        return self.num_samples


class WithSizeSampler(Sampler):

    def __init__(self, data_source, shuffle=False):
        if not hasattr(data_source, 'imgsize_at'):
            raise ValueError("To use WithSizeSample, the data_source must implement imgsize_at function")
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.data_source)).long()
        else:
            indices = range(len(self.data_source))
        for i in indices:
            yield (i, self.data_source.imgsize_at(i))

    def __len__(self):
        return len(self.data_source)


class GroupingBatchSampler(object):

    def __init__(self, sampler, batch_size, grouping):
        self.sampler = sampler
        self.batch_size = batch_size
        self.grouping = grouping

    def __iter__(self):
        batch = {}
        for idx, info in self.sampler:
            group = self.grouping(info)
            if not group in batch:
                batch[group] = [idx,]
            else:
                batch[group].append(idx)
            if len(batch[group]) == self.batch_size:
                yield batch[group]
                batch[group] = []
        for group, b in batch.items():
            if len(b) > 0:
                yield batch

    def __len__(self):
        raise NotImplementedError("Not supported")


def group_by_ratio(expected_sizes):

    def grouping(size):
        best_portion = 0
        h, w = size
        for expected_size in expected_sizes:
            th, tw = expected_size
            scaling = min(w / tw, h / th)
            ch, cw = int(th * scaling), int(tw * scaling)
            portion = (ch * cw) / (w * h)
            if portion > best_portion:
                best_size = expected_size
                best_portion = portion
        return best_size

    return grouping