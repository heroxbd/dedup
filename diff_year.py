# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:30:33 2018

@author: zhuoj
"""

#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()

import pandas as pd, itertools as it, h5py, numpy as np
# ^^^ command line specification

from collections import Counter
au = pd.read_csv(args.ipt)


# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())

# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

# this is expanded to be used with keywords as well
maxyear = max(au[args.field])
minyear = min(au[args.field])

dl = (abs(al[1]-bl[1])/(maxyear-minyear)  
    for (al, bl) in it.combinations(au.groupby('id')[args.field],2))
x = np.array(list(dl), dtype='u2')

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('c_{}'.format(args.field), data=x, compression="gzip", shuffle=True)
