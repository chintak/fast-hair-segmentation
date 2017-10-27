import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import configs
import multiprocessing as mp
import numpy as np
import os
from skimage.io import imread
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb

import data
import featurizer
from configs import HAIR, FACE, BKG
import configs


def train(model_fname, model_type, dtrain, dval=None, dtest=None, cont=None):
    params = getattr(configs, model_type)()
    evallist = []
    if dval: evallist.append((dval, 'val'))
    if dtest: evallist.append((dtest, 'test'))
    evallist.append((dtrain, 'train'))
    bst = xgb.train(params, dtrain, params['num_round'], evallist,
                    xgb_model=cont, early_stopping_rounds=5)
    bst.save_model(model_fname)
    bst.dump_model('dump.raw.txt')


def test(model_fname, model_type, fset):
    params = getattr(configs, model_type)(val=True)
    bst = xgb.Booster(params=params)
    bst.load_model(model_fname)
    Xset, yset = load_svmlight_file(fset)
    dset = xgb.DMatrix(Xset, label=yset)
    pset = bst.predict(dset)
    yset = label_binarize(yset, classes=[0, 1, 2])
    cm = confusion_matrix(np.argmax(yset, 1), np.argmax(pset, 1), labels=[0, 1, 2])
    dump_latex(cm)
    plot_prcurve(yset, pset, '{}_{}_pr.png'.format(
        model_fname.split('.')[0], fset.split('.')[-1]))
    yset = (yset.argmax(1) == 0).astype(np.uint8)
    pset = (pset.argmax(1) == 0).astype(np.uint8)
    print 'HAIR class accuracy:', accuracy_score(yset, pset)


def dump_latex(cm):
    print '\\begin{{bmatrix}}{}&{}&{}\\\\{}&{}&{}\\\\{}&{}&{}\\end{{bmatrix}}'.format(
        *cm.ravel())


def plot_prcurve(yset, pset, name):
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    # For each class
    n_classes = yset.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(yset[:, i],
                                                            pset[:, i])
        average_precision[i] = average_precision_score(yset[:, i], pset[:, i])

    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i])   )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig(name, frameon=False, bbox_inches='tight')


def get_model_feature_type(name):
    n, e = os.path.splitext(os.path.basename(name))
    mtype, ftype = n.split('_')[:2]
    if not hasattr(configs, mtype):
        print 'Invalid model type in {}. See configs.py'.format(name)
        mtype, ftype = None, None
    if not hasattr(configs, ftype):
        print 'Invalid feature type in {}. See configs.py'.format(name)
        mtype, ftype = None, None
    return mtype, ftype


def args():
    args = argparse.ArgumentParser()
    args.add_argument('model', help='if --train is specified, then the name of'
                      ' the trained model; else the model name for testing')
    args.add_argument('-c', '--cont', help='model file to resume training')
    args.add_argument('-t', '--train', help='train set file')
    args.add_argument('-v', '--validation', help='validation set file')
    args.add_argument('-e', '--test', help='test set file')
    return args.parse_args()


def main():
    parse = args()
    mtype, ftype = get_model_feature_type(parse.model)
    if not hasattr(configs, mtype):
        import sys
        print 'Invalid model type. See configs.py'
        sys.exit(0)
    print 'Model type: {}\nFeature type: {}\nModel name: {}'.format(
        mtype, ftype, parse.model)

    if parse.train:
        dtrain = xgb.DMatrix(parse.train) if parse.train else None
        dval = xgb.DMatrix(parse.validation) if parse.validation else None
        dtest = xgb.DMatrix(parse.test) if parse.test else None
        train(parse.model, mtype,
              dtrain, dval=dval, dtest=dtest,
              cont=parse.cont)
    elif parse.validation:
        test(parse.model, mtype, fset=parse.validation)
    elif parse.test:
        test(parse.model, mtype, fset=parse.test)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    print "Done"
