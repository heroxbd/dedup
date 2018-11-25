#!/usr/bin/env python
import argparse, json
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', nargs="+", help="input")
psr.add_argument('--field', default="org", help="field input")
args = psr.parse_args()

import pandas as pd, re, os
from glob import glob
from functools import reduce
fl = args.ipt

def org_bag(fn):
    au=pd.read_csv(fn)
    k=os.path.basename(fn)[:-4]

    rk = k.replace('_','\W+')
    sau = au[au.name.str.contains(re.compile("^{}$".format(rk), re.IGNORECASE))]
    cluster = sau.groupby('org')['id'].apply(list)
    return k, list(cluster.values)

def uniglue_bag(fn):
    au=pd.read_csv(fn)
    k=os.path.basename(fn)[:-4]

    cluster=au.groupby("group_result")['id'].apply(list)
    return k, list(cluster.values)

if args.field == "org":
    func = org_bag
elif args.field == "uniglue":
    func = uniglue_bag
    
rst = dict([func(fn) for fn in fl])
json.dump(rst, open(args.opt, 'w'))
