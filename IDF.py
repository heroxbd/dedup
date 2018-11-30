#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', nargs="+", help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()

import pandas as pd
import numpy as np
from collections import Counter
import h5py
# ^^^ command line specification

wordlist = pd.read_csv(args.ipt)

# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())

# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

# this is expanded to be used with keywords as well
wordlist['count'] = np.ones((len(wordlist)))
wordcount = wordlist.set_index([args.field, "id"]).count(level=args.field)
IDF = len(wordlist)/wordcount

IDF.to_csv(args.opt, index=False)

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('IDF', data=IDF)
