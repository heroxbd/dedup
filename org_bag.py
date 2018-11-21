#!/usr/bin/env python
import argparse, json
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
args = psr.parse_args()

import pandas as pd, re, os
from glob import glob
from functools import reduce
fl = glob("{}/*.csv".format(args.ipt))

def org_bag(fn):
    au=pd.read_csv(fn)
    k=os.path.basename(fn)[:-4]
    rk = k.replace('_','\W+')    
    sau = au[au.name.str.contains(re.compile("^{}$".format(rk), re.IGNORECASE))]
    cluster = sau.groupby('org')['id'].apply(list)
    return k, list(cluster.values)

rst = dict([org_bag(fn) for fn in fl])
json.dump(rst, open(args.opt, 'w'))
