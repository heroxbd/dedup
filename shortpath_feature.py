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

os.chdir('D:/jike paradis')
data = pd.read_csv('./data/bin_yu.csv')

wname =  pd.value_counts(data['name'])
name = wname.index
wid =  pd.value_counts(data['id'])
id = wid.index

MisNodename = pd.DataFrame({'node':name,'weight':wname})
MisNodename = MisNodename.drop('Bin Yu')
MisNodeID = pd.DataFrame({'node':id,'weight':wid})
MisNodes = MisNodename.append(MisNodeID)

Mislinks = pd.DataFrame(columns=['source','target','weight'])
namelist = dict(list(data.groupby('name')))
del(namelist['Bin Yu'])


namelist100 = dict(list(data.groupby('name'))[0:100])
del(namelist100['Bin Yu'])

for name0 in namelist.keys():
    paperlist = namelist.get(name0)
    linklist1 = pd.DataFrame({'source':paperlist['id'],'target':name0,'weight':1})
    Mislinks = Mislinks.append(linklist1)

'''
for name1 in namelist100.keys():
    for name2 in namelist100.keys():
        conum = 0
        i = 0
        if (name1 < name2):
            conum = len([i for i in namelist.get(name1)['id'].values if i in namelist.get(name2)['id'].values])
            if conum!= 0:
                linklist2 = np.array((name1,name2,conum))
                i = i+1
                Mislinks.loc[i] = linklist2
                #Mislinks = Mislinks.append(linklist2,ignore_index=True)          
'''


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



x = np.array(())
for (al, bl) in it.combinations(data['id'],2):
    try:
        m = 1/nx.shortest_path_length(G, source=al, target=bl, weight=None)
        x = np.append(x,m)
    except:
        x = np.append(x,0)


# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('shortpath', data=x, compression="gzip", shuffle=True)


