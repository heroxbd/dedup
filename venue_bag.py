#!/usr/bin/env python
import argparse, json
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
args = psr.parse_args()

pv = json.load(open(args.ipt))

import pandas as pd
def venue_bag(v):
    pr = pd.DataFrame.from_dict([{'id':x['id'], 'venue':x['venue']} for x in v])
    cluster = pr.groupby('venue')['id'].apply(list)
    return list(cluster.values)
rst = {k: venue_bag(pv[k]) for k in pv}
json.dump(rst, open(args.opt, 'w'))
