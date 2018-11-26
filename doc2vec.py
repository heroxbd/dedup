# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 15:04:58 2018

This script generates a doc2vec model from the training/validating dataset.
The input is a LIST of csv file path, while each containig fields 'id', 'title', and 'abstract'.
In each record, 'title' and 'abstract' will be merged into a 'paragraph'.
The LIST of 'paragraph' becomes the corpus for doc2vec training.
After training, the doc2vec model convert a paragraph into a vector.  
The doc2vec model is saved to 'd2v.model' file.

@author: HZQ
"""

# === import all the dependencies ============================================

import argparse
psr = argparse.ArgumentParser("word2vec feature engineer")
psr.add_argument("-i", nargs = '+', default=['data/train/ia.csv'], dest='ipt', help="input")
psr.add_argument("-o", default='features/d2v_singlet.model', dest='opt', help="output")
args = psr.parse_args()


import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
#from sklearn.manifold import TSNE
#import csv
import time
import datetime
#import os

# === Functions ===============================================================
def print_log(lp, log_str):
    local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    log_str = '['+local_time+'] '+log_str
    print(log_str,end='')
    
    with open(lp,'a') as f:
        f.write(log_str)


# === Specify major parameters ================================================
input_file_name_list = args.ipt
input_latent_space_dimension = 50
input_training_epochs = 100 # default 100
initial_learning_rate = 0.050 # default 0.025
learning_rate_dec_rate = 0.0004 # default 0.00002

log_path = ('doc2vec_training.log') # path/name of the log file

# === load files =============================================================
print_log(log_path, '\n\n\n=== NEW RUN ===\n\n')

# start timing
time_start = time.time()
time_tmp = time.time()

# load file with panda module
reader = pd.concat([pd.read_csv(fn) for fn in input_file_name_list])
print_log(log_path, 'csv file opened...\n')

# === Data preparation =======================================================
# delete duplicating records
reader_distinct = reader.drop_duplicates('id')

# prepare raw corpus & id_list
corpus = []
id_list = []
for num in range(len(reader_distinct)):
    corpus.append( str(reader_distinct.iloc[num]['title']) + str(reader_distinct.iloc[num]['abstract']) )
    id_list.append( reader_distinct.iloc[num]['id'] )

# fill the empty record with 'nan'
data = [] # Training data should be a LIST of str
for idx in range(len(corpus)):
    if type(corpus[idx]) == str:
        data.append(corpus[idx])
    else:
        data.append('nan')

print_log(log_path,'Preparing data (tokenizing)...\n')
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[id_list[i]]) for i, _d in enumerate(data)]

print_log(log_path,'Data preparation: %.3f s.\n' % (time.time()-time_tmp))
time_tmp = time.time()

# === Train the doc2vec model ================================================
print_log(log_path, 'latent_space_dimension: {0}\n'.format(input_latent_space_dimension))

# transfering parameters
max_epochs = input_training_epochs
vec_size = input_latent_space_dimension
alpha = initial_learning_rate

print_log(log_path,'Loading model...\n')
model = Doc2Vec(vector_size = vec_size,
                alpha = alpha, 
                workers = 8,
                min_alpha = 0.00025,
                min_count = 1,
                dm = 1)

model.build_vocab(tagged_data)

# start training
print_log(log_path,'Training...\n')

for epoch in range(max_epochs):
    
    # --- log string ---
    epoch_log_str = 'iter. {0} '.format(epoch+1)
    epoch_log_str = epoch_log_str + ', learn_rate %.5f ' % model.alpha
    # --- training ---
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs) #old expression
    # decrease the learning rate
    model.alpha -= learning_rate_dec_rate
    # --- log string ---
    docvec_zero = model.docvecs[id_list[0]]
    epoch_log_str = epoch_log_str + ', vec0_norm %.3f ' % np.linalg.norm(docvec_zero)
    epoch_log_str = epoch_log_str + ', vec0_var %.3f\n' % docvec_zero.var()
    print_log(log_path, epoch_log_str)
    
# --- log ---
print_log(log_path,'Training finished, time spent: '+str(datetime.timedelta(seconds=time.time()-time_tmp))+'\n')
time_tmp = time.time()

# === save doc2vec to file ====================================================
model_file_name = args.opt
model.save(model_file_name)
# --- log ---
print_log(log_path,"Doc2Vec model Saved.\n")
