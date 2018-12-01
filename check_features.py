#!/usr/bin/env python2
import os
import os.path as osp
import numpy as np
import h5py
import argparse
import json
import time
import glob
from collections import OrderedDict
from scipy.stats import pearsonr

parser = argparse.ArgumentParser(description='check features')

parser.add_argument('--split', type=str, default='validate',
                    help='train on split: train, validate')
args = parser.parse_args()

# Find features
feat_file_list = glob.glob('features/' + args.split + '/*.h5')
exclude_filename = ['label', 'id_pairs', 'valid_index']
for f in exclude_filename:
    f = osp.join('features', args.split, f + '.h5')
    if f in feat_file_list:
        feat_file_list.remove(f)
feature_ids = [os.path.split(f)[1][:-3] for f in feat_file_list]

# Load label
with h5py.File('features/' + args.split + '/label.h5', 'r') as f:
    label = f['label'][:].astype(np.int)
    print('label shape:')
    print(label.shape)

for feat_id, feat_file in zip(feature_ids, feat_file_list):
    print('Loading features ' + feat_id)
    time_start = time.time()
    with h5py.File(feat_file, 'r') as f:
        feat = f[feat_id][:]
        # if features are saved as numpy structure
        for field in feat.dtype.names:
            print('field: ' + field)
            feat_field = feat[field].ravel()
            print('pearson corelation: %.6f' % pearsonr(label, feat_field)[0])

