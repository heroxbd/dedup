# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:23:22 2018

This script will make the pair features given a author's csv file.
features include:
    distance, norm(vec1 - vec2)
    angle, angle between vec1 and vec2, in rad
    text_length_multiply = length1 * length2
    valid = 1 if (lengh of both texts > 10) otherwise 0.
the output file will be saved in /features/authorName_doc2vec.h5

@author: HZQ
"""

#!/usr/bin/env python


# === import all the dependencies ============================================
'''
import argparse
psr = argparse.ArgumentParser("baseline solution")
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input")
psr.add_argument('--field', default='org', help="the field to count common entries in")
args = psr.parse_args()
'''

import pandas as pd
import itertools as it
import h5py
import numpy as np
from collections import Counter
from gensim.models.doc2vec import Doc2Vec
import os

# === Specify major parameters ================================================
#au = pd.read_csv(args.ipt)
item_file_name = 'ia.csv' # the id-title-abstract data at /data/ia.csv
input_file_name = 'li_ma.csv' # the file is read at /data/*.csv
output_file_name = 'li_ma_doc2vec.h5' # the file will be save at /features/*.h5


'''
# (original codes)

# the central fucntion is sum((Counter(al[1]) & Counter(bl[1])).values())
# it counts the common org of a and b including duplications.  For
# example, if a has 3 "Tsinghua" and b has 2, the common org is
# counted as 2.

# this is expanded to be used with keywords as well


dl = (sum((Counter(al[1]) & Counter(bl[1])).values())
      for (al, bl) in it.combinations(au.groupby('id')[args.field],2))
x = np.array(list(dl))
'''

# === import data =============================================================
# load file with panda module
reader = pd.read_csv(os.path.join('data',item_file_name))
# delete duplicating records
reader_distinct = reader.drop_duplicates('id')

# prepare raw corpus & id_list
corpus = []
id_list = []
for num in range(len(reader_distinct)):
    corpus.append( str(reader_distinct.iloc[num]['title']) + str(reader_distinct.iloc[num]['abstract']) )
    id_list.append( reader_distinct.iloc[num]['id'] )

au = pd.read_csv(os.path.join('data',input_file_name)) # author associated file
model = Doc2Vec.load('d2v.model')

# === calculate similarities ==================================================

dl = [];
total_num = len(au)*(len(au)-1)/2
progress = 0
progress_step = 0.01
count = 0
tag_a_cache=[]
for (al, bl) in it.combinations(au.groupby('id')['id'],2):
    tag_a = al[0]
    tag_b = bl[0]
    if tag_a == tag_a_cache:
        pass
    else:
        idx_a = id_list.index(tag_a)
    idx_b = id_list.index(tag_b)    
    #idx_a = id_list.index(tag_a)
    #idx_b = id_list.index(tag_b)
    docvec_a = model.docvecs[str(idx_a)]
    docvec_b = model.docvecs[str(idx_b)]
    norm_a = np.linalg.norm(docvec_a)
    norm_b = np.linalg.norm(docvec_b)
    distance = np.linalg.norm(docvec_a - docvec_b)
    angle = np.arccos( max( min( np.dot(docvec_a,docvec_b)/(norm_a*norm_b), 1), -1) )
    text_length_multiply = len(corpus[idx_a])*len(corpus[idx_b])
    if len(corpus[idx_a])>10 and len(corpus[idx_b])>10:
        valid = 1
    else:
        valid = 0
    
    dl.append( [distance, angle, text_length_multiply, valid] )
    
    count = count + 1;
    if count/total_num > progress:
        print('current progress: ' + str(progress*100) + ' percent.\n')
        progress = progress + progress_step

print(len(dl))
x = np.array(list(dl))

'''
# (original codes)output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('c_{}'.format(args.field), data=x, compression="gzip", shuffle=True)
'''
with h5py.File(os.path.join('features',output_file_name), 'w') as opt:
    opt.create_dataset('doc2vecdata', data=x, compression="gzip", shuffle=True)