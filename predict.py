import argparse
import cPickle as pickle
from glob import glob
import multiprocessing as mp
import numpy as np
import os
from scipy.io import loadmat
from skimage.io import imread
from sklearn.datasets import dump_svmlight_file
from subprocess import call
import sys
import xgboost as xgb
import tempfile as tm

import configs
from configs import HAIR, FACE, BKG
import data
from utils import *

EPS = np.finfo(float).eps


def pr_calc(yt, yp):
    tp = np.sum((yt == yp) & (yt == 1))
    tn = np.sum((yt == yp) & (yt == 0))
    fp = np.sum((yt != yp) & (yt == 1))
    fn = np.sum((yt != yp) & (yt == 0))
    return tp, tn, fp, fn


def evaluate(names, keyps, model_fname, q):
    models = hr_name_to_models(model_fname)
    ttp, ttn, tfp, tfn = 0, 0, 0, 0
    for k, (name, keyp) in enumerate(zip(names, keyps)):
        if not os.path.exists(name): continue
        im = imread(name)
        pr = hr_predict_single(im, keyp, models, overlap=0.5)
        gt = data.img2gt(name)
        tp, tn, fp, fn = pr_calc((gt==HAIR), (pr==HAIR))
        ttp += tp; ttn += tn; tfp += fp; tfn += fn
        # if k % 50 == 0: print "[{}] Done {}".format(os.getpid(), k)
    q.put((ttp, ttn, tfp, tfn))


def eval(model_fname, mat_viz_file):
    print "=================================="
    q = mp.Queue()

    names, keypoints = data.mat_to_name_keyp(mat_viz_file)

    NUM_TRAIN_SAMPS = len(names)
    nprocs = mp.cpu_count()
    chunksize = int(NUM_TRAIN_SAMPS // nprocs)
    procs = []

    for i in range(nprocs):
        lim = chunksize * (i+1) if i < nprocs - 1 else NUM_TRAIN_SAMPS
        p = mp.Process(target=evaluate,
                       args=(names[chunksize*i:lim],
                             keypoints[chunksize*i:lim],
                             model_fname, q))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    ttp, ttn, tfp, tfn = 0., 0., 0., 0.
    for i in range(nprocs):
        tp, tn, fp, fn = q.get()
        ttp += tp; ttn += tn; tfp += fp; tfn += fn

    print "Model: {} pixel level:".format(model_fname)
    print "\thair accuracy = {:.03f}".format(1. - (tfp + tfn) / (EPS + tfn + tfp + ttp + ttn))
    print "\tprec \t= {:.03f}".format((ttp) / (EPS + ttp + tfp))
    print "\trec  \t= {:.03f}".format((ttp) / (EPS + ttp + tfn))


def args():
    args = argparse.ArgumentParser()
    args.add_argument('model_file', help='')
    args.add_argument('mat_file', help='')
    return args.parse_args()


if __name__ == '__main__':
    parse = args()
    eval(parse.model_file, parse.mat_file)
