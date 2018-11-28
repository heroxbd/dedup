#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
psr.add_argument('--field', default='year', help="the field to count common entries in")
args = psr.parse_args()
# ^^^ command line specification

import pandas as pd, itertools as it, h5py, numpy as np

au = pd.read_csv(args.ipt, index_col=0)

yearspan = au[args.field].max() - au[args.field].min()
cau = au.drop_duplicates().sort_values(by='id')

dl = [float(abs(a-b))/yearspan
      for (a, b) in it.combinations(cau[args.field].values,2)]
x = np.array(dl, dtype=[('diff_year', 'f4')])

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('diff_year', data=x, compression="gzip", shuffle=True)
