import argparse
import configs
import multiprocessing as mp
import numpy as np
import os
from skimage.io import imread
import xgboost as xgb

import data
import featurizer
from configs import HAIR, FACE, BKG
import configs


def train(model_fname, dtrain, dval=None, dtest=None, cont=None):
    params = configs.basic()
    evallist = []
    if dval: evallist.append((dval, 'val'))
    if dtest: evallist.append((dtest, 'test'))
    evallist.append((dtrain, 'train'))
    bst = xgb.train(params, dtrain, params['num_round'], evallist,
                    xgb_model=cont, early_stopping_rounds=5)
    bst.save_model(model_fname)
    bst.dump_model('dump.raw.txt')


def test():
    pass


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
    mode_train = True if parse.train else False
    dtrain = xgb.DMatrix(parse.train) if parse.train else None
    dval = xgb.DMatrix(parse.validation) if parse.validation else None
    dtest = xgb.DMatrix(parse.test) if parse.test else None

    if dtrain:
        train(parse.model, dtrain, dval=dval, dtest=dtest, cont=parse.cont)
    elif dtest or dval:
        test()
