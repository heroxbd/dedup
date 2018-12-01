#!/usr/bin/env python
#import csv
#import os
import pandas as pd
import numpy as np
import itertools as it
from matplotlib import pylab as plt
import h5py
import networkx as nx

import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-i", default = 'features/validate/c_org/long_wang.h5', dest='ipt', help="input")
psr.add_argument("-p", default = 'features/validate/id_pairs/long_wang.h5', dest='ipt_id_pair', help="input")
psr.add_argument("--field", default = 'org_jaccard_similarity_metric', dest='field', help="input")
psr.add_argument("-o", default = 'features/validate/shortpath_c_org/long_wang.h5',dest='opt', help="output")
args = psr.parse_args()

input_file_distance_pair = args.ipt
input_file_id_pair = args.ipt_id_pair
input_record_array_field_name = args.field

# --- read id_pair -----------------------------
with h5py.File(input_file_id_pair, 'r') as ipt_id_pair:
    idpairlist = ipt_id_pair['id_pairs'][:]

id_pair_list_a = []
id_pair_list_b = []
id_pair_list = []
for idx in range(len(idpairlist)):
    id_pair_list_a.append(idpairlist[idx][0])
    id_pair_list_b.append(idpairlist[idx][1])

nodes_list_unique = list(np.unique(id_pair_list_a+id_pair_list_b))
nodes_list_unique_idx = list(np.arange(len(nodes_list_unique)))


# --- read pair distance ------------------------
with h5py.File(input_file_distance_pair, 'r') as ipt_pair_dist:
    tmp_field_name_list = list(ipt_pair_dist.keys())
    distlist = ipt_pair_dist[tmp_field_name_list[0]][:]

distance_list = []
edge_list_clean = []
for idx in range(len(distlist)):
    tmp = float(distlist[idx][input_record_array_field_name])
    if tmp > 0:
        distance_list.append(1/tmp)
        edge_list_clean.append([nodes_list_unique.index(id_pair_list_a[idx]), nodes_list_unique.index(id_pair_list_b[idx]),{'weight': 1/tmp}])
    else:
        distance_list.append(0)


# --- graph computation -------------------------

G=nx.Graph()
G.add_nodes_from(nodes_list_unique_idx)
G.add_edges_from(edge_list_clean)
graphs = list(nx.connected_component_subgraphs(G))

list_of_graphs_node_dict = []
for idx in range(len(graphs)):
    list_of_graphs_node_dict.append(graphs[idx]._node)

#print( nx.shortest_path_length(G, source = 0, target = 1, weight = 'weight')) # testing code

dist = []
count = 0
total_num = len(id_pair_list_a)
progress = 0
progress_step = 0.02
for idx in range(len(id_pair_list_a)):
    
    id_a = id_pair_list_a[idx]
    id_b = id_pair_list_b[idx]
    idx_a = nodes_list_unique.index(id_a)
    idx_b = nodes_list_unique.index(id_b)
    
    for idx in range( len(list_of_graphs_node_dict) ):
        if list_of_graphs_node_dict[idx].__contains__(idx_a):
            subgraph_containing_idx_a = idx
        else:
            pass
        if list_of_graphs_node_dict[idx].__contains__(idx_b):
            subgraph_containing_idx_b = idx
        else:
            pass
        
    if subgraph_containing_idx_a == subgraph_containing_idx_b:
        dist_tmp = nx.shortest_path_length(graphs[subgraph_containing_idx_a], source = idx_a, target = idx_b, weight = 'weight')
        if dist_tmp > 0:
            dist.append(1/dist_tmp)
        else:
            dist.append(0.0)
    else:
        dist.append(0.0)
    
    # show progress
    count = count + 1;
    if count/total_num > progress:
        print( 'current progress: %.1f percent.' % (count/total_num*100) )
        progress = progress + progress_step


# --- output file ---------------------------------------
dsn = args.opt.split('/')[-2] # doc2vec_singlet_native
x = np.array(dist, dtype=[('{}_distance'.format(dsn), 'f4')])

# output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset(dsn, data=x, compression="gzip", shuffle=True)
