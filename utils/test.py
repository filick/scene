#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:46:48 2017

@author: wayne
"""

import ClassAwareSampler

rand_cyc_iter = ClassAwareSampler.RandomCycleIter([1, 2, 3])
print([next(rand_cyc_iter) for _ in range(10)])

cls_data_list = [list() for _ in range(10)]