#!/usr/bin/env python2
import os
import os.path as osp
import numpy as np
import pandas as pd
import h5py
import random
import argparse
import json
import time
import copy
import glob
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.externals import joblib


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
    parser.add_argument('--bootstrap', action='store_true',
                        help='bootstrap in data preparation')
    parser.add_argument('--train_split', type=str, default='validate',
                        help='train on split: train, validate')
    parser.add_argument('--tune_hyper', action='store_true',
                        help='tune hyper-parameters')
    parser.add_argument('--tune_split', type=str, default='validate',
                        help='tune on split: train, validate')
    parser.add_argument('--eval', action='store_true',
                        help='eval models')
    parser.add_argument('--eval_split', type=str, default='validate_val',
                        help='eval on split: train_val, validate_val')
    parser.add_argument('--predict', action='store_true',
                        help='use models to predict on set')
    parser.add_argument('--predict_split', type=str, default='train',
                        help='predict on split: train, train_val, validate, validate_val, test')
    parser.add_argument('--retrain', action='store_true',
                        help='retrain all models')
    parser.add_argument('--model_params', type=str, default=None,
                        help='model params of XGB')
    # parser.add_argument('--remove_missing', action='store_true',
    #                     help='remove samples with missing data in train')
    # parser.add_argument('--name_split_file', type=str, default='data/split_1fold.json',
    #                     help='file that contains train and val splits of names')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='ratio of train names to all names')
    parser.add_argument('--random_state', type=int, default=2018,
                        help='random state for XGB and sklearn')
    args = parser.parse_args()
    return args


class BootstrapSplit():
    def __init__(self, init_split):
        self.init_split = copy.copy(init_split)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.init_split.get_n_splits()

    def split(self, X, y, groups=None):
        splits = list(self.init_split.split(X, y))
        for train, test in splits:
            train_neg = np.where(y[train] == 0)[0]
            train_pos = np.where(y[train] == 1)[0]
            train_neg = np.random.choice(train_neg, len(train_pos))
            train = np.concatenate((train_pos, train_neg))
            np.random.shuffle(train)
            yield train, test


def tune_hyper(args):
    # Define parameters
    tune_params = {'n_estimators': [50]}
    fixed_params = {'learning_rate': 0.1,
                    'scale_pos_weight': 1,
                    'random_state': args.random_state,
                    'n_jobs': 4}

    # Load data
    data, label, _ = loaders(args, args.tune_split)
    data_train, label_train, data_val, label_val = data['train'], label['train'], data['val'], label['val']

    # Define CV split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=5, train_size=args.train_ratio,
                                 random_state=args.random_state)
    if args.bootstrap:
        sss = BootstrapSplit(sss)
    # from sklearn.model_selection import ShuffleSplit
    # sss = ShuffleSplit(n_splits=5, train_size=args.train_ratio, 
    #                              random_state=args.random_state)
    # from sklearn.model_selection import KFold
    # sss = KFold(n_splits=5, shuffle=True, random_state=args.random_state)

    # Conduct CV
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, classification_report
    clf = GridSearchCV(XGBClassifier(**fixed_params),
                       param_grid=tune_params, 
                       scoring=make_scorer(f1_score),
                       n_jobs=1, refit=False,
                       cv=sss, verbose=1, pre_dispatch='n_jobs')
    clf.fit(data_train, label_train)

    # Print result
    print("Grid scores on development set:")
    print('')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print('')
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print('')

    fixed_params.update(clf.best_params_)
    # Path to save models
    if not osp.exists('models'):
        os.makedirs('models')
    # Save model
    model_filename = 'models/XGB_tune.param'
    json.dump(fixed_params, open(model_filename, 'w'))
    print('Best model params saved to ' + model_filename)
    args.model_params = model_filename
    args.nb_samples = 100000
    args.model_ids = ['XGB']
    args.retrain = True
    train(args)


def f1_score(gt, pred):
    correctly_pred = np.logical_and(gt, pred).sum()
    total_pred = np.sum(pred)
    total_gt = np.sum(gt)
    if total_gt == 0 or total_pred == 0:
        f1 = 0
    else:
        precision = float(correctly_pred) / total_pred
        recall = float(correctly_pred) / total_gt
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
    print('Precision: %.4f, recall: %.4f' % (precision, recall))
    return f1


def loaders(args, split):
    print('Loading features')
    # Load features
    if 'train' in split:
        load_split = 'train'
    elif 'validate' in split:
        load_split = 'validate'
    else:
        load_split = split
    if len(args.feature_ids) == 0:
        feat_file_list = glob.glob('features/' + load_split + '/*.h5')
        exclude_filename = ['label', 'id_pairs', 'valid_index']
        for f in exclude_filename:
            f = osp.join('features', load_split, f + '.h5')
            if f in feat_file_list:
                feat_file_list.remove(f)
        args.feature_ids = [os.path.split(f)[1][:-3] for f in feat_file_list]
    else:
        feat_file_list = ['features/' + load_split + '/' + f + '.h5' for f in args.feature_ids]
    print('Using features: ' + ' '.join(args.feature_ids))
    data = []
    feat_names = []
    sep_data = None
    for feat_id, feat_file in zip(args.feature_ids, feat_file_list):
        print('Loading features ' + feat_id)
        time_start = time.time()
        with h5py.File(feat_file, 'r') as f:
            feat = f[feat_id][:]
            # if features are saved as numpy structure
            for field in feat.dtype.names:
                feat_names.append(field)
                feat_field = feat[field]
                # for concatenate, add axis 1
                if len(feat_field.shape) == 1:
                    feat_field = feat_field[:, np.newaxis]
                data.append(feat_field)
            if sep_data is None:
                sep_data = f['sep'][:]
            else:
                assert np.all(sep_data == f['sep'][:]), 'sep of %s not the same with previous seps' % feat_id
        print('%.2fs have passed' % (time.time() - time_start))
    data = np.concatenate(data, axis=1)

    # For train_val, load labels
    if (not args.predict) or (split in ['train_val', 'validate_val']):
        print('Loading labels')
        with h5py.File('features/' + load_split + '/label.h5', 'r') as f:
            label = f['label'][:]
            sep = f['sep'][:]
        assert data.shape[0] == label.shape[0], 'lengths of feature and label not equal'
        assert np.all(sep == sep_data), 'sep of label not the same with sep of features'
        sep = np.concatenate([[0], sep])
        with open(load_split + '_names.mk', 'r') as f:
            names_all = f.readlines()[0].strip().split('=')[1].split()
        names_all = sorted(names_all)
        index = []
        for i, name in enumerate(names_all):
            index.append({'name':name, 'start':sep[i], 'end':sep[i+1]})

        # Split names into train and val split
        name_split_file = osp.join('data', load_split, 'split_1fold.json')
        if osp.exists(name_split_file):
            name_split = json.load(open(name_split_file))
            names_train, names_val = sorted(name_split['train']), sorted(name_split['val'])
        else:
            raise AssertionError('run python sample_seed.py first!')
            # if load_split == 'validate':
            #     assert len(names_all) == 49, 'validate names not equal to 49!'
            # random.seed(274)
            # nb_val = int(np.floor(len(names_all) * (1 - args.train_ratio)))
            # val_index = random.sample(range(len(names_all)), nb_val)
            # train_index = np.setdiff1d(range(len(names_all)), val_index)
            # names_train = [names_all[i] for i in train_index]
            # names_val = [names_all[i] for i in val_index]
            # name_split = {'train':names_train, 'val':names_val}
            # with open(name_split_file, 'w') as f:
            #     json.dump(name_split, f)

        # Split data into train and val
        index_train = [np.arange(ind['start'], ind['end']) for ind in index if ind['name'] in names_train]
        index_train = np.concatenate(index_train)
        index_val = [np.arange(ind['start'], ind['end']) for ind in index if ind['name'] in names_val]
        index_val = np.concatenate(index_val)
        print('Loaded #train: %d, #val: %d' % (len(index_train), len(index_val)))
            
        # Resample step 1: filter out samples that are all zeros
        # if (not args.eval) and (not args.predict):
        #     index_train = index_train[np.any(data[index_train] != 0, axis=1)]
        #     index_val = index_val[np.any(data[index_val] != 0, axis=1)]
        #     print('keep only samples with nonzero features: train %d, val %d' % (len(index_train), len(index_val)))

        # Resample step 2: #train = #val = args.nb_samples
        if args.nb_samples > 0:
            if args.nb_samples < len(index_train):
                index_train_pos = np.random.choice(index_train[label[index_train] == 1], args.nb_samples / 2)
                index_train_neg = np.random.choice(index_train[label[index_train] == 0], args.nb_samples - args.nb_samples / 2)
                index_train = np.concatenate((index_train_pos, index_train_neg))
                # index_train = np.random.choice(index_train, args.nb_samples)
            if args.nb_samples < len(index_val):
                index_val_pos = np.random.choice(index_val[label[index_val] == 1], args.nb_samples / 2)
                index_val_neg = np.random.choice(index_val[label[index_val] == 0], args.nb_samples - args.nb_samples / 2)
                index_val = np.concatenate((index_val_pos, index_val_neg))
                # index_val = np.random.choice(index_val, args.nb_samples)
        data = OrderedDict((('train', data[index_train]), ('val', data[index_val])))
        label = OrderedDict((('train', label[index_train]), ('val', label[index_val])))
        print('Feature size: train ' + str(data['train'].shape) + ', val ' + str(data['val'].shape))
        print('Label size: train %d, val %d' % (len(label['train']), len(label['val'])))
    else:
        print('Feature size: %d, %d' % data.shape)

    if args.predict:
        if split in ['train_val', 'validate_val']:
            data = data['val']
            names = names_val
            sep = [0]
            for ind in index:
                if ind['name'] in names_val:
                    sep.append(ind['end'] - ind['start'] + sep[-1])
            sep = sep[1:]
        else:
            sep = sep_data
            names = sorted(json.load(open('data/pubs_' + load_split + '.json')).keys())
        return data, sep, names
    else:
        return data, label, feat_names


def train(args):
    ''' 
    TODO:
    how to mine hard data: read adaboost
    ensemble several random forests
    '''
    # Load data
    data, label, feat_names = loaders(args, args.train_split)
    data_train, label_train, data_val, label_val = data['train'], label['train'], data['val'], label['val']

    # Initialize classifier
    models = OrderedDict()
    for model_id in args.model_ids:
        if model_id == 'RandomForest':
            models['RandomForest'] = RandomForestClassifier(class_weight='balanced')
        elif 'XGB' in model_id:
            fixed_params = {'max_depth': 5,
                            'subsample': 1,
                            'gamma': 0,
                            'min_child_weight': 2,
                            'n_estimators': 100,
                            'learning_rate': 0.1,
                            'scale_pos_weight': 1,
                            'random_state': args.random_state,
                            'n_jobs': 4}
            if args.model_params is not None:
                fixed_params.update(json.load(open(args.model_params)))
            print(fixed_params)
            models['XGB'] = XGBClassifier(**fixed_params)
            # models['XGB'] = XGBClassifier(scale_pos_weight=1)
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
        f1[model_id] = f1_score(label_val, (pred_val > 0.5).ravel())
        print('F1 score: %.6f' % f1[model_id])
        # for ensemble
        preds.append(pred_val)
        # save model
        joblib.dump(model, model_filename)
        print('Model saved to ' + model_filename)
        # save feature importance
        if model_id == 'XGB':
            pd.Series(data=model.feature_importances_, index=feat_names).to_csv('models/XGB_feat_importance.csv')

    # Ensemble
    if len(args.model_ids) < 2:
        return
    preds = np.concatenate(preds, axis=1)
    if args.ensemble == 'mean':
        preds = (preds.mean(axis=1) > 0.5).ravel()
    else:
        raise ValueError('ensemble strategy %s not implemented' % args.ensemble)
    print('ensemble %s F1 score: %.6f' % (args.ensemble, f1_score(label_val, preds)))


def evaluate(args):
    # Load data
    args.nb_samples = -1 # no resampling in evaluation
    data, label, _ = loaders(args, args.eval_split)
    data, label = data['val'], label['val']
    # data, label = data['train'], label['train']

    # Eval
    print('Evaluation starts')
    time_start = time.time()
    preds = []
    for model_id in args.model_ids:
        model_filename = osp.join('models', model_id + '.model')
        model = joblib.load(model_filename)
        pred = model.predict_proba(data)[:, 1][:, np.newaxis]
        print('%s f1 score: %.6f' % (model_id, f1_score(label, (pred > 0.5).ravel())))
        preds.append(pred)
    # Ensemble
    if len(args.model_ids) < 2:
        return
    preds = np.concatenate(preds, axis=1)
    if args.ensemble == 'mean':
        preds = (preds.mean(axis=1) > 0.5).ravel()
    else:
        raise ValueError('ensemble strategy %s not implemented' % args.ensemble)
    print('ensemble %s F1 score: %.6f' % (args.ensemble, f1_score(label, preds)))
    print('%.2fs have passed' % (time.time() - time_start))


def predict(args):
    # Load data
    args.nb_samples = -1 # no resampling in prediction
    args.remove_missing = False # no remove data in prediction
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
        preds = preds[0].ravel()
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
    retrain_flag = np.any([not osp.exists(osp.join('models', model_id + '.model')) for model_id in args.model_ids])
    if args.retrain:
        print('Train on ' + args.train_split + ' split')
        train(args)
    elif args.tune_hyper:
        print('Tune hyper-parameters on ' + args.tune_split + ' split')
        tune_hyper(args)
    elif args.eval:
        assert retrain_flag == False, 'Not all models are trained'
        print('Eval on ' + args.eval_split + ' split')
        evaluate(args)
    elif args.predict:
        assert retrain_flag == False, 'Not all models are trained'
        print('Predict on ' + args.predict_split + ' split')
        predict(args)
    else:
        print('Nothing to do')
