# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 00:51:22 2018

@author: zhuoj
"""
import csv
import os
import pandas as pd
import numpy as np
import itertools as it
from matplotlib import pylab as plt
import h5py

import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="author input")
args = psr.parse_args()

data = pd.read_csv(args.ipt)

wname =  pd.value_counts(data['name'])
name = wname.index
wid =  pd.value_counts(data['id'])
id = wid.index

MisNodename = pd.DataFrame({'node':name,'weight':wname})
an = os.path.basename(args.ipt)[:-4].replace('_', ' ')
MisNodename = MisNodename.drop(an)
MisNodeID = pd.DataFrame({'node':id,'weight':wid})
MisNodes = MisNodename.append(MisNodeID)

Mislinks = pd.DataFrame(columns=['source','target','weight'])
namelist = dict(list(data.groupby('name')))
del(namelist[an])

for name0 in namelist.keys():
    paperlist = namelist.get(name0)
    linklist1 = pd.DataFrame({'source':paperlist['id'],'target':name0,'weight':1})
    Mislinks = Mislinks.append(linklist1)

end = 0
for i in range(len(data)):
    start = end
    if (i>0 and data['id'][i-1]!=data['id'][i]):
        end = i
        paper = data[start:end]
        for name1 in paper['name']:
            for name2 in paper['name']:
                if (name1 < name2):
                    linklist2 = np.array((name1,name2,1))
                    Mislinks.loc[i] = linklist2


import networkx as nx
G=nx.Graph()
G.add_nodes_from(MisNodes)
G.add_edges_from(list(Mislinks[['source', 'target']].to_records(index=False)))
graphs = list(nx.connected_component_subgraphs(G))


idnum = len(np.unique(data['id']))
pairlength = idnum*(idnum-1)
x = np.zeros(pairlength)
i = 0
for (al, bl) in it.combinations(np.unique(data['id']),2):
    i1 = [i1 for i1 in range(len(graphs)) if al in graphs[i1]]
    i2 = [i2 for i2 in range(len(graphs)) if bl in graphs[i2]]
    if i1==i2:
        x[i]=1
        i = i+1

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('shortpath', data=x, compression="gzip", shuffle=True)
