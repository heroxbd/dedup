#!/usr/bin/env python
import argparse, json
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
args = psr.parse_args()

pv = json.load(open(args.ipt))
rst = {k:[[x['id'] for x in pv[k]]] for k in pv}
json.dump(rst, open(args.opt, 'w'))
