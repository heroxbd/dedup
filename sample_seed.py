#!/usr/bin/env python
import glob
import os
import h5py
import json
import random
import numpy as np

# Calculate lens and labels
labels = sorted(glob.glob('features/validate/label/*.h5'))
names = [os.path.split(l)[1][:-3] for l in labels]
nb_names = len(names)
nb_val = int(np.floor(nb_names * 0.2))
nb_labels = []
nb_pos = []
for i, (name, labelf) in enumerate(zip(names, labels)):
    with h5py.File(labelf, 'r') as f:
        label = f['label'][:]
        nb_labels.append(len(label))
        nb_pos.append(sum(label))
        rate = float(nb_pos[i]) / nb_labels[i]
        print('%s: %.4f' % (name, rate))
nb_labels = np.array(nb_labels)
nb_pos = np.array(nb_pos)

# Try different splits
diff_pos_rate = []
for i in range(500):
    random.seed(i)
    val_index = random.sample(range(nb_names), nb_val)
    train_index = np.setdiff1d(range(nb_names), val_index)
    d = float(nb_pos[train_index].sum()) / nb_labels[train_index].sum() - \
        float(nb_pos[val_index].sum()) / nb_labels[val_index].sum()
    diff_pos_rate.append(d)

# Print result
seed = np.argmin(np.abs(diff_pos_rate))
print('Best seed: %d' % seed)
random.seed(seed)
val_index = random.sample(range(nb_names), nb_val)
train_index = np.setdiff1d(range(nb_names), val_index)
d1 = float(nb_pos[train_index].sum()) / nb_labels[train_index].sum()
d2 = float(nb_pos[val_index].sum()) / nb_labels[val_index].sum()
print('train pos rate: %.4f' % d1)
print('val pos rate: %.4f' % d2)

# Save names
name_split_file = 'data/validate/split_1fold.json'
names_splitted = {'train':[names[i] for i in train_index],
                  'val':[names[i] for i in val_index]}
with open(name_split_file, 'w') as f:
    json.dump(names_splitted, f)
print('result saved to ' + name_split_file)