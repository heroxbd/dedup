import argparse
import json
import numpy as np
import time
# import sklearn.metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate train f1 score')

    parser.add_argument('input', type=str, help='predicion filename')
    parser.add_argument('--names', type=str, default='', help='names to calculate f1')
    parser.add_argument('--average', type=str, default='binary', help='manner to calculate f1')
    args = parser.parse_args()

    return args

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
    return f1

def evaluate(input, names=None, average='binary'):
    '''
    input could be a list or a json filename, 
    names assign the names to calculate f1, default use all names
    average: binary - pairwise f1 for positive samples only
             macro - mean f1 of positive and negative samples
    assume that one paper only assign to one author within the same name
    '''
    if isinstance(input, str) and input.split('.')[-1] == 'json':
        pred = json.load(open(input))
    else:
        assert isinstance(input, list), 'input must be a list or a json filename'

    print('Evaluation starts with %s manner.' % average)
    time_start = time.time()
    # Load gt file
    label = json.load(open('data/assignment_train.json'))
    if names is None:
        names = label.keys()

    # Find pairs of each name
    pairs_gt_all = []
    pairs_pred_all = []
    nb_papers_all = 0
    for name in names:
        # Create pairs classification of gt
        authors_gt = label[name]
        authors_pred = pred[name]
        papers_gt = [p for papers in authors_gt for p in papers]
        nb_papers = len(papers_gt)
        nb_papers_all += nb_papers
        pairs_gt = np.zeros((nb_papers, nb_papers), dtype=np.bool)
        count = 0
        for author in authors_gt:
            nb_papers_auth = len(author)
            pairs_gt[count:count+nb_papers_auth, count:count+nb_papers_auth] = 1
            count += nb_papers_auth

        # Create pairs classification of prediction
        pairs_pred = np.zeros((nb_papers, nb_papers), dtype=np.bool)
        # index papers in authors_pred
        pp2auth = np.zeros(nb_papers, dtype=np.int32)
        for i, paper in enumerate(papers_gt):
            for j, papers_pred in enumerate(authors_pred):
                if paper in papers_pred:
                    pp2auth[i] = j 
                    papers_pred.remove(paper)
                    break
        # find papers with same author id, assign those elements in pairs_pred to be 1
        for auth in range(len(authors_pred)):
            auth_same = np.where(pp2auth == auth)[0]
            nb_auth_same = len(auth_same)
            pairs_pred[np.repeat(auth_same, nb_auth_same), 
                       np.tile(auth_same, nb_auth_same)] = 1
        # extract triu of pairs_gt and pairs_pred, then flatten
        pairs_gt = pairs_gt[np.triu_indices(nb_papers, 1)]
        pairs_pred = pairs_pred[np.triu_indices(nb_papers, 1)]
        # append pairs to all pairs
        pairs_gt_all.append(pairs_gt)
        pairs_pred_all.append(pairs_pred)
        
    pairs_gt_all = np.concatenate(pairs_gt_all)
    pairs_pred_all = np.concatenate(pairs_pred_all)
    if average == 'binary':
        # 'binary': Only report results for the class specified by pos_label
        f1 = f1_score(pairs_gt_all, pairs_pred_all)
    elif average == 'macro':
        # 'macro': Calculate metrics for each label, and find their unweighted mean. 
        # This does not take label imbalance into account.
        # for pos_label
        f1 = f1_score(pairs_gt_all, pairs_pred_all)
        # for neg_label
        more_neg = nb_papers_all * (nb_papers_all - 1) // 2 - len(pairs_gt_all)
        pairs_gt_all = np.logical_not(pairs_gt_all)
        pairs_pred_all = np.logical_not(pairs_pred_all)
        correctly_pred = np.logical_and(pairs_gt_all, pairs_pred_all).sum()
        total_pred = np.sum(pairs_pred_all) + more_neg
        total_gt = np.sum(pairs_gt_all) + more_neg
        if total_gt == 0 or total_pred == 0:
            f1 += 0
        else:
            precision = float(correctly_pred) / total_pred
            recall = float(correctly_pred) / total_gt
            if precision == 0 or recall == 0:
                f1 += 0
            else:
                f1 += 2 * precision * recall / (precision + recall) 
        f1 /= 2
    else:
        raise ValueError('manner %s not implemented' % average)

    print('Evaluation takes %.2fs' % (time.time() - time_start))
    return f1


if __name__ == '__main__':
    args = parse_args()
    if len(args.names) == 0:
        args.names = None
    f1 = evaluate(args.input, args.names, args.average)
    print('f1 score on train set: %.6f' % f1)
