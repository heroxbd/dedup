#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()

import pandas as pd, itertools as it, h5py, numpy as np
# ^^^ command line specification

au = pd.read_csv(args.ipt)


# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())

# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

# this is expanded to be used with keywords as well
maxyear = max(au[args.field])
minyear = min(au[args.field])
idlist = np.unique(au['id'])
difflist = []


for (aid, bid) in it.combinations(idlist,2):
    ayear = np.array((au[au['id']==aid]['year']))[0]
    byear = np.array((au[au['id']==bid]['year']))[0]
    difflist.append(abs(ayear-byear)/(maxyear-minyear))

x = np.array(list(difflist), dtype='u2')

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('diff year', data=x, compression="gzip", shuffle=True)
