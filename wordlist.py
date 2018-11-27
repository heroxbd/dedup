#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()

import pandas as pd
# ^^^ command line specification

au = pd.read_csv(args.ipt)

# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())

# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

# this is expanded to be used with keywords as well
stringlist = au[['id', args.field]]

rst = pd.concat([pd.DataFrame({"id": r['id'],"wordcut":
                               pd.Series(r[args.field].split(' '))}) for (i, r) in
                 stringlist.iterrows()])

rst.to_csv(args.opt, index=False)
