# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:23:22 2018

This script will make the pair features given a author's csv file.
features include:
    distance, norm(vec1 - vec2)
    angle, angle between vec1 and vec2, in rad
    sqrt_text_length_multiply = sqrt( length1 * length2 )
the output file will be saved in /features/authorName_doc2vec.h5

@author: HZQ
"""

#!/usr/bin/env python

# === import all the dependencies ============================================

import argparse
psr = argparse.ArgumentParser("word2vec feature engineer")
psr.add_argument("-i", default='data/train/item/li_ma.csv', dest='ipt', help="input")
psr.add_argument("-m", default='features/d2v_singlet.model', dest='model_path', help="input")
psr.add_argument("-o", default='features/train/doc2vec_singlet_native/li_ma.h5', dest='opt', help="output")
args = psr.parse_args()

import pandas as pd
import itertools as it
import h5py
import numpy as np
from collections import Counter
from gensim.models.doc2vec import Doc2Vec
import os

# === Specify major parameters ================================================
#au = pd.read_csv(args.ipt)
item_file_name = 'data/train/ia.csv' # the id-title-abstract data at /data/ia.csv
input_file_path = args.ipt # the file is read at /data/*.csv
output_file_path = args.opt # the file will be save at /features/*.h5
model_path = args.model_path

(csv_filepath,csv_filename) = os.path.split(input_file_path)
(author_name,csv_extension) = os.path.splitext(csv_filename)
print('the author name is: {}'.format(author_name))

'''
# (original codes in c_org.py)

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
reader = pd.read_csv(item_file_name)
# delete duplicating records
#reader_distinct = reader.drop_duplicates('id')
reader_author = reader.loc[reader['auid'] == author_name]

# prepare raw corpus & id_list
corpus_author = []
id_list_author = []
for num in range(len(reader_author)):
    corpus_author.append( str(reader_author.iloc[num]['title']) + str(reader_author.iloc[num]['abstract']) )
    id_list_author.append( reader_author.iloc[num]['id'] )

au = pd.read_csv(input_file_path) # author associated file
model = Doc2Vec.load(model_path)

# === calculate similarities ==================================================

dl = [];
total_num = len(au)*(len(au)-1)/2
progress = 0
progress_step = 0.05
count = 0
tag_a_cache=[]
for (al, bl) in it.combinations(au.groupby('id')['id'],2):
    tag_a = al[0]
    tag_b = bl[0]
    if tag_a == tag_a_cache:
        pass
    else:
        idx_a = id_list_author.index(tag_a)
    idx_b = id_list_author.index(tag_b)    
    #idx_a = id_list.index(tag_a)
    #idx_b = id_list.index(tag_b)
    docvec_a = model.docvecs[tag_a]
    docvec_b = model.docvecs[tag_b]
    norm_a = np.linalg.norm(docvec_a)
    norm_b = np.linalg.norm(docvec_b)
    distance = np.linalg.norm(docvec_a - docvec_b)
    angle = np.arccos( max( min( np.dot(docvec_a,docvec_b)/(norm_a*norm_b), 1), -1) )
    text_length_multiply = len(corpus_author[idx_a])*len(corpus_author[idx_b])
    
    #if len(corpus_author[idx_a])>10 and len(corpus_author[idx_b])>10:
    #    valid = 1
    #else:
    #    valid = 0
    
    dl.append( [distance, angle, np.sqrt(text_length_multiply)] )
    
    count = count + 1;
    if count/total_num > progress:
        print( 'current progress: %.2f percent.' % (count/total_num*100) )
        progress = progress + progress_step

print(len(dl))
x = np.array(list(dl))

'''
# (original codes in c_ort.py)output .h5:
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('c_{}'.format(args.field), data=x, compression="gzip", shuffle=True)
'''
with h5py.File(output_file_path, 'w') as opt:
    opt.create_dataset('doc2vecdata', data=x, compression="gzip", shuffle=True)