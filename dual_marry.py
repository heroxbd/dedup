#!/usr/bin/env python
import pandas as pd, argparse, os
psr = argparse.ArgumentParser("hunter for duplicates")
psr.add_argument("-o", dest='opt', help="output of duplicates")
psr.add_argument('ipt', help="input of authors")
args = psr.parse_args()

au = pd.read_csv(args.ipt)
nm = os.path.basename(args.ipt).replace(".csv", "")

iv = au.query('name=="{}"'.format(nm.replace("_", " ")))
iva = iv['id'].values

from collections import Counter
cv = Counter(iva) - Counter(set(iva))

cva = (pd.Series(cv) + 1)/2
cva.to_csv(args.opt)
