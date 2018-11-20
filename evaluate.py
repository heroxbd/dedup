import argparse
import json
import numpy as np
import sklearn.metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate train f1 score')

    parser.add_argument('input', type=str, help='predicion filename')
    parser.add_argument('--names', type=str, default='', help='names to calculate f1')
    args = parser.parse_args()

    return args

def evaluate(input, names=None):
    '''
    input could be a list or a json filename, 
    names assign the names to calculate f1, default use all names
    assume that one paper only assign to one author within the same name
    '''
    if isinstance(input, str) and input.split('.')[-1] == 'json':
        pred = json.load(open(input))
    else:
        assert isinstance(input, list), 'input must be a list or a json filename'

    # Load gt file
    label = json.load(open('data/assignment_train.json'))
    if names is None:
        names = label.keys()

    # Caculate f1 scores of each name
    f1 = []
    for name in names:
        # Create pairs classification of gt
        authors_gt = label[name]
        authors_pred = pred[name]
        papers_gt = [p for papers in authors_gt for p in papers]
        nb_papers = len(papers_gt)
        pairs_gt = np.zeros((nb_papers, nb_papers), dtype=np.int32)
        count = 0
        for author in authors_gt:
            nb_papers_auth = len(author)
            pairs_gt[count:count+nb_papers_auth, count:count+nb_papers_auth] = 1
            count += nb_papers_auth

        # Create pairs classification of prediction
        pairs_pred = np.zeros((nb_papers, nb_papers), dtype=np.int32)
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
        # calculate f1
        f1.append(sklearn.metrics.f1_score(pairs_gt, pairs_pred, 
                  average='macro'))

    return np.mean(f1)


if __name__ == '__main__':
    args = parse_args()
    if len(args.names) == 0:
        args.names = None
    f1 = evaluate(args.input, args.names)
    print('f1 score on train set: %.6f' % f1)