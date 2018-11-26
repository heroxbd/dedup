#!/usr/bin/env python
'''
Read assignment_*.json and output grand truth column.
'''

import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="id ordering, should be something under item/")
psr.add_argument('--ref', default="data/assignment_train.json", help="truth cluster")
args = psr.parse_args()

import pandas as pd, itertools as it, h5py, numpy as np, json, os

iv = pd.read_csv(args.ipt)['id'].unique()

# Implicit assumption: infer name from input filename.
nm = os.path.basename(args.ipt).replace(".csv", "")
at=json.load(open(args.ref))
lm=at[nm]
d = pd.concat([pd.DataFrame({"id":v, "seq":i}) for i, v in enumerate(lm)])

# short circuit with the first elements.
x = np.array([(al[1].values[0]==bl[1].values[0]) or (np.intersect1d(al[1].values,bl[1].values).size>0)
              for (al,bl) in it.combinations(d.groupby('id')['seq'], 2)])

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('label', data=x, compression="gzip", shuffle=True)
