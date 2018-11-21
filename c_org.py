#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
args = psr.parse_args()

import pandas as pd, itertools as it, h5py, numpy as np
from collections import Counter
au = pd.read_csv(args.ipt)

# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())

# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

dl = ((al[0], bl[0], sum((Counter(al[1]) & Counter(bl[1])).values()))
      for (al, bl) in it.combinations(au.groupby('id')['org'],2))
x = np.array(list(dl), dtype=[('id1', 'S24'), ('id2', 'S24'), ('c_org', 'u2')])

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('c_org', data=x, compression="gzip", shuffle=True)
