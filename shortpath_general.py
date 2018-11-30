import csv
import os
import pandas as pd
import numpy as np
import itertools as it
from matplotlib import pylab as plt
import h5py
'''
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="author input")
args = psr.parse_args()
'''

os.chdir('D:/jike paradis/data')

with h5py.File('bin_yu.h5', 'r') as ipt:
    idlist = ipt['id_pairs'][:]

    
idpairs = pd.concat(idlist['id1'],idlist['id2'])
MisNodes = np.unique(idlist)

MisLinks = pd.concat(idpairs,feature)
MisLinks = MisLinks[(True-MisLinks['feature'].isin([0]))]


import networkx as nx
G=nx.Graph()
G.add_nodes_from(MisNodes)
G.add_edges_from(list(MisLinks[['source', 'target']].to_records(index=False)))
graphs = list(nx.connected_component_subgraphs(G))

x = []
for (al, bl) in it.combinations(np.unique(data['id']),2):
    i1 = [i1 for i1 in range(len(graphs)) if al in graphs[i1]]
    i2 = [i2 for i2 in range(len(graphs)) if bl in graphs[i2]]
    if i1==i2 and i1 != []:
        m = 1/nx.shortest_path_length(graphs[i1[0]], source=al, target=bl, weight=None)
        x.append(m)
    else:
        x.append(0)

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('shortpath', data=x, compression="gzip", shuffle=True)
