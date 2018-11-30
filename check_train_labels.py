#!/usr/bin/env python2
import h5py
import numpy as np
import os
import glob
import json

names = sorted(json.load(open('data/assignment_train.json')))
res = {}
for name in names:
    with h5py.File('features/train/label/' + name + '.h5', 'r') as f:
        label = f['label'][:].astype(np.bool).ravel()
    with h5py.File('output/train/' + name + '.h5', 'r') as f:
        pred = f['prediction'][:].astype(np.bool).ravel()
    acc = (label == pred).sum() / float(len(label))
    # print(name + 'acc: %.6f' % acc)
    res[name] = acc
# TODO here, sort acc
keys = sorted(res.keys(), lambda x:res[x])
for key in keys:
    print(key + 'acc: %.6f' % res[key])