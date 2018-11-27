#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()

import pandas as pd, numpy as np
# ^^^ command line specification

au = pd.read_csv(args.ipt)

# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())

# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

# this is expanded to be used with keywords as well
stringlist = au[['id', args.field]]

def dfo(r):
    if type(r[args.field]) is float:
        # !!emtpy field!!
        # insert the id so that it does not intersect with any other.
        st = [r['id']]
    else:
        st = pd.Series(r[args.field].split(' '))
    return pd.DataFrame({"id": r['id'], args.field: st})

rst = pd.concat([dfo(r) for (i, r) in stringlist.iterrows()])

rst.to_csv(args.opt, index=False)
