#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', nargs="+", help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()

import pandas as pd, numpy as np

wordlist = pd.concat([pd.read_csv(f) for f in args.ipt])

# # this is expanded to be used with keywords as well
wordlist['count'] = np.ones((len(wordlist)))
wordcount = wordlist.set_index([args.field, "id"]).count(level=args.field)
IDF = len(wordlist)/wordcount

IDF.to_csv(args.opt, index=False)
