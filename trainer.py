import argparse
import configs
import multiprocessing as mp
import numpy as np
import os
from skimage.io import imread
from sklearn.datasets import load_svmlight_file
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
    nset = load_svmlight_file(fset)
    dset = xgb.DMatrix(nset)
    preds = bst.predict(dset)
    pass


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


if __name__ == '__main__':
    parse = args()
    mtype, ftype = get_model_feature_type(parse.model)
    if not hasattr(configs, mtype):
        import sys
        print 'Invalid model type. See configs.py'
        sys.exit(0)
    print 'Model type {} & Model name {}'.format(mtype, parse.model)
    dtrain = xgb.DMatrix(parse.train) if parse.train else None
    dval = xgb.DMatrix(parse.validation) if parse.validation else None
    dtest = xgb.DMatrix(parse.test) if parse.test else None

    if dtrain:
        train(parse.model, mtype,
              dtrain, dval=dval, dtest=dtest,
              cont=parse.cont)
    elif dtest or dval:
        test(parse.model, mtype, dval=dval, dtest=dtest)
