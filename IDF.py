#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', nargs="+", help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()

import pandas as pd, numpy as np
from collections import Counter

wordlist = pd.concat([pd.read_csv(f) for f in args.ipt])

# this is expanded to be used with keywords as well
wordcount = Counter(wordlist[args.field].values)
IDF = float(len(wordlist))/np.array(tuple(wordcount.values()))
rst = pd.DataFrame({"IDF":IDF}, index=tuple(wordcount.keys()))
rst.to_csv(args.opt)
