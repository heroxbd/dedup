#!/usr/bin/env python
import os
import os.path as osp
import numpy as np
import h5py
import argparse
import json
import time
import glob
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='Classifier for pairwise AND')
    
    parser.add_argument('--model_ids', nargs='+', default=['RandomForest', 'XGB'], 
                        help='list of model to use')
    parser.add_argument('--feature_ids', nargs='+', default=[], 
                        help='list of features to use, default use all c_*.h5 under features/train')
    parser.add_argument('--ensemble', type=str, default='mean',
                        help='ensemble strategy')
    parser.add_argument('--nb_samples', type=int, default=-1,
                        help='#samples used in train and val, -1 for all')
    parser.add_argument('--eval', action='store_true',
                        help='eval models')
    parser.add_argument('--predict', action='store_true',
                        help='use models to predict on set')
    parser.add_argument('--predict_split', type=str, default='train',
                        help='predict on split: train, train_val, val, test')
    parser.add_argument('--retrain', action='store_true',
                        help='retrain all models')
    parser.add_argument('--name_split_file', type=str, default='data/split_1fold.json',
                        help='file that contains train and val splits of names')
    parser.add_argument('--name_split_train_ratio', type=float, default=0.8,
                        help='ratio of train names to all names')
    args = parser.parse_args()
    return args


def loaders(args, split):
    print('Loading features')
    # Load features
    load_split = 'train' if 'train' in split else split
    if len(args.feature_ids) == 0:
        feat_file_list = glob.glob('features/' + load_split + '/*.h5')
        exclude_filename = ['label', 'id_pairs']
        for f in exclude_filename:
            f = osp.join('features', load_split, f + '.h5')
            if f in feat_file_list:
                feat_file_list.remove(f)
        args.feature_ids = [os.path.split(f)[1][:-3] for f in feat_file_list]
    else:
        feat_file_list = ['features/' + load_split + '/' + f + '.h5' for f in args.feature_ids]
    print('Using features: ' + ' '.join(args.feature_ids))
    data = []
    sep_data = None
    for feat_id, feat_file in zip(args.feature_ids, feat_file_list):
        print('Loading features ' + feat_id)
        time_start = time.time()
        with h5py.File(feat_file, 'r') as f:
            feat = f[feat_id][:]
            if len(feat.dtype) > 1:
                # if features are saved as numpy structure
                for field in feat.dtype.names:
                    feat_field = feat[field]
                    # for concatenate, add axis 1
                    if len(feat_field.shape) == 1:
                        feat_field = feat_field[:, np.newaxis]
                    data.append(feat_field)
            else:
                # for concatenate, add axis 1
                if len(feat.shape) == 1:
                    feat = feat[:, np.newaxis]
                data.append(feat)
            if sep_data is None:
                sep_data = f['sep'][:]
            else:
                assert np.all(sep_data == f['sep'][:]), 'sep of %s not the same with previous seps' % feat_id
        print('%.2fs have passed' % (time.time() - time_start))
    data = np.concatenate(data, axis=1)

    # For train_val, load labels
    if split == 'train_val':
        print('Loading labels')
        with h5py.File('features/train/label.h5', 'r') as f:
            label = f['label'][:]
            sep = f['sep'][:]
        assert data.shape[0] == label.shape[0], 'lengths of feature and label not equal'
        # assert np.all(sep == sep_data), 'sep of label not the same with sep of features'
        sep = np.concatenate([[0], sep])
        names_trainset = json.load(open('data/assignment_train.json'))
        names_trainset = sorted(names_trainset.keys())
        index = []
        for i, name in enumerate(names_trainset):
            index.append({'name':name, 'start':sep[i], 'end':sep[i+1]})

        # Split names into train and val split
        if osp.exists(args.name_split_file):
            name_split = json.load(open(args.name_split_file))
            names_train, names_val = name_split['train'], name_split['val']
        else:
            nb_names_train = int(np.ceil(len(names_trainset) * args.name_split_train_ratio))
            names_train, names_val = names_trainset[:nb_names_train], names_trainset[nb_names_train:]
            name_split = {'train':names_train, 'val':names_val}
            with open(args.name_split_file, 'w') as f:
                json.dump(name_split, f)

        # Split data into train and val
        index_train = [np.arange(ind['start'], ind['end']) for ind in index if ind['name'] in names_train]
        index_train = np.concatenate(index_train)
        index_val = [np.arange(ind['start'], ind['end']) for ind in index if ind['name'] in names_val]
        index_val = np.concatenate(index_val)
        # Resample step 1: filter out samples that are all zeros
        if (not args.eval) and (not args.predict):
            index_train = index_train[np.any(data[index_train] != 0, axis=1)]
            index_val = index_val[np.any(data[index_val] != 0, axis=1)]
            print('keep only samples with nonzero features: train %d, val %d' % (len(index_train), len(index_val)))
        # Resample step 2: #train = #val = args.nb_samples
        if args.nb_samples > 0:
            if args.nb_samples < len(index_train):
                index_train_pos = np.random.choice(index_train[label[index_train] == 1], args.nb_samples / 2)
                index_train_neg = np.random.choice(index_train[label[index_train] == 0], args.nb_samples - args.nb_samples / 2)
                index_train = np.concatenate((index_train_pos, index_train_neg))
            if args.nb_samples < len(index_val):
                index_val_pos = np.random.choice(index_val[label[index_val] == 1], args.nb_samples / 2)
                index_val_neg = np.random.choice(index_val[label[index_val] == 0], args.nb_samples - args.nb_samples / 2)
                index_val = np.concatenate((index_val_pos, index_val_neg))
        data = OrderedDict((('train', data[index_train]), ('val', data[index_val])))
        label = OrderedDict((('train', label[index_train]), ('val', label[index_val])))
        print('Feature size: train ' + str(data['train'].shape) + ', val ' + str(data['val'].shape))
        print('Label size: train %d, val %d' % (len(label['train']), len(label['val'])))
    else:
        print('Feature size: %d, %d' % data.shape)

    if args.predict:
        if split == 'train_val':
            sep = sep[1:]
            sep = sep[len(names_train):] - sep[len(names_train) - 1]
            data = data['val']
            names = names_val
        else:
            # TODO for val and test split
            sep = sep_data
            names = None
        return data, sep, names
    else:
        return data, label


def train(args):
    ''' 
    TODO:
    how to mine hard data: read adaboost
    ensemble several random forests
    '''
    # Load data
    data, label = loaders(args, 'train_val')
    data_train, label_train, data_val, label_val = data['train'], label['train'], data['val'], label['val']

    # Initialize classifier
    models = OrderedDict()
    for model_id in args.model_ids:
        if model_id == 'RandomForest':
            models['RandomForest'] = RandomForestClassifier(class_weight='balanced')
        elif model_id == 'XGB':
            models['XGB'] = XGBClassifier(scale_pos_weight=1)
        else:
            raise ValueError('model %s not implemented' % model_id)
    # Path to save models
    if not osp.exists('models'):
        os.makedirs('models')

    # Training starts
    print('Training starts')
    f1 = OrderedDict()
    preds = []
    for model_id, model in models.iteritems():
        model_filename = osp.join('models', model_id + '.model')
        if osp.exists(model_filename) and not args.retrain:
            continue 
        # train
        print('Training model %s ...' % model_id)
        time_start = time.time()
        model.fit(data_train, label_train)
        print('Training finished. %.2fs passed' % (time.time() - time_start))
        # validate
        pred_val = model.predict_proba(data_val)[:, 1][:, np.newaxis]
        f1[model_id] = f1_score(label_val, pred_val > 0.5, average='binary')
        print('F1 score: %.6f' % f1[model_id])
        # for ensemble
        preds.append(pred_val)
        # save model
        joblib.dump(model, model_filename)
        print('Model saved to ' + model_filename)

    # Ensemble
    if len(args.model_ids) < 2:
        return
    preds = np.concatenate(preds, axis=1)
    if args.ensemble == 'mean':
        preds = (preds.mean(axis=1) > 0.5)
    else:
        raise ValueError('ensemble strategy %s not implemented' % args.ensemble)
    print('ensemble %s F1 score: %.6f' % (args.ensemble, f1_score(label_val, preds, average='binary')))


def evaluate(args):
    # Load data
    args.nb_samples = -1 # no resampling in evaluation
    data, label = loaders(args, 'train_val')
    data, label = data['val'], label['val']

    # Eval
    print('Evaluation starts')
    time_start = time.time()
    preds = []
    for model_id in args.model_ids:
        model_filename = osp.join('models', model_id + '.model')
        model = joblib.load(model_filename)
        pred = model.predict_proba(data)[:, 1][:, np.newaxis]
        print('%s f1 score: %.6f' % (model_id, f1_score(label, pred > 0.5, average='binary')))
        preds.append(pred)
    # Ensemble
    if len(args.model_ids) < 2:
        return
    preds = np.concatenate(preds, axis=1)
    if args.ensemble == 'mean':
        preds = (preds.mean(axis=1) > 0.5)
    else:
        raise ValueError('ensemble strategy %s not implemented' % args.ensemble)
    print('ensemble %s F1 score: %.6f' % (args.ensemble, f1_score(label, preds, average='binary')))
    print('%.2fs have passed' % (time.time() - time_start))


def predict(args):
    # Load data
    args.nb_samples = -1 # no resampling in prediction
    data, sep, names = loaders(args, args.predict_split)

    # Predict
    print('Prediction starts')
    time_start = time.time()
    preds = []
    for model_id in args.model_ids:
        model_filename = osp.join('models', model_id + '.model')
        model = joblib.load(model_filename)
        preds.append(model.predict_proba(data)[:, 1][:, np.newaxis])
    # Ensemble
    if len(args.model_ids) >= 2:
        preds = np.concatenate(preds, axis=1)
        if args.ensemble == 'mean':
            preds = preds.mean(axis=1)
        else:
            raise ValueError('ensemble strategy %s not implemented' % args.ensemble)
    else:
        preds = preds[0]
    print('%.2fs have passed' % (time.time() - time_start))
    
    # Path to save result
    output_dir = osp.join('output', args.predict_split)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    sep = np.concatenate(([0], sep))
    for i, name in enumerate(names):
        predict_file = osp.join(output_dir, name + '.h5')
        with h5py.File(predict_file, 'w') as f:
            f.create_dataset('prediction', data=preds[sep[i]:sep[i+1]], compression="gzip", shuffle=True)
        print('Prediction of ' + name + ' saved to ' + predict_file)


if __name__ == '__main__':
    args = parse_args()
    retrain_flag = not np.all([osp.exists(osp.join('models', model_id + '.model')) for model_id in args.model_ids])
    if args.retrain or retrain_flag:
        train(args)
    elif args.eval:
        assert retrain_flag == False, 'Not all models are trained'
        evaluate(args)
    elif args.predict:
        assert retrain_flag == False, 'Not all models are trained'
        predict(args)
    else:
        print('Nothing to do')