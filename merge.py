#!/usr/bin/env python

import h5py, argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', nargs="+", help="input")
psr.add_argument('--field', default='org', help="the field to merge")
args = psr.parse_args()

with h5py.File(args.ipt[0]) as ipt:
    dtp = ipt[args.field].dtype

with h5py.File(args.opt, "w") as raw:
    pair = raw.create_dataset(args.field, shape=(1e12,), dtype=dtp, compression="gzip", shuffle=True)
    p = 0
    for f in args.ipt:
        with h5py.File(f) as ipt:
            s = ipt[args.field].size
            pair[p:p+s] = ipt[args.field]
            p += s
    pair.resize((p,))
