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

iv = pd.read_csv(args.ipt)['id'].values

# Implicit assumption: infer name from input filename.
nm = os.path.basename(args.ipt).replace(".csv", "")
at=json.load(open(args.ref))
lm=at[nm]
d = pd.concat([pd.DataFrame({"id":v, "seq":i}) for i, v in enumerate(lm)]).set_index('id')

dl = (a==b for (a,b) in it.combinations(d.loc[iv].values, 2))
x = np.array(list(dl))

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('label', data=x, compression="gzip", shuffle=True)
