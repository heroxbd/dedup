# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 03:03:06 2018

@author: zhuoj
"""

#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()

import pandas as pd, h5py, numpy as np
# ^^^ command line specification

au = pd.read_csv(args.ipt)

# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())

# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

# this is expanded to be used with keywords as well
x = au[args.field]
wordlist = []
for i in range(len(x)):
    wordlist.append( [word for word in x.split(' ')])

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('wordlist'.format(args.field), data=wordlist, compression="gzip", shuffle=True)