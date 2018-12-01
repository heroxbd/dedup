#!/usr/bin/env python
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-i", default='D:/jike paradis/data/c_c_wang.csv', dest='ipt', help="input")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('--field', default='venue', help="the field to count common entries in")
#psr.add_argument('--idf', default='data/train/venue_idf.csv', help="idf feature generator")
psr.add_argument('--idf', default='D:/jike paradis/data/venue_idf.csv', help="idf feature generator")
args = psr.parse_args()

import pandas as pd, itertools as it, h5py, numpy as np

# ^^^ command line specification


from collections import Counter
au = pd.read_csv(args.ipt)
IDFinput = pd.read_csv(args.idf, index_col = 0)

# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())

# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

# this is expanded to be used with keywords as well


def f(al,bl):
    commondict = Counter(al[1]) & Counter(bl[1])
    TF = 1/(len(al[1])*len(bl[1]))
    overlap = sum(commondict.values())
    sumlength = len(al[1]) + len(bl[1])
    if commondict == Counter():
        IDF = 0
    else:
        IDF = IDFinput.loc[list(commondict.keys())]
        IDF = sum(IDF['IDF']*np.array(list(commondict.values())))
    return overlap, sumlength, IDF, TF

dl = (f(al,bl)
      for (al, bl) in it.combinations(au.groupby('id')[args.field],2))
x = np.array(list(dl), dtype='f4')

df = pd.DataFrame({'{}_overlap'.format(args.field): x[:,0], 
                   '{}_share_dummy'.format(args.field): x[:,0]!=0,
                   '{}_jaccard_similarity_metric'.format(args.field): x[:,0].astype('float32')/(x[:,1]-x[:,0]),
                   '{}_logIDF'.format(args.field): np.log(x[:,2]),
                   '{}_logTFIDF'.format(args.field): np.log(x[:,2])*np.log(x[:,3])
                   }
)

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('c_{}'.format(args.field), data=df.to_records(index=False), compression="gzip", shuffle=True)
