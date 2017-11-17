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
from utils import *

Conf = None
Set = ''
WINDOW = None
DATA_DIR = 'LFW/'
tmdir = tm.mkdtemp(dir=DATA_DIR)


def gen_train_data(proc_names, proc_keypoints, hr_model_configs=[],
                   out_q=None, seed=1234):
    if not Conf or not WINDOW: raise ValueError('Conf not set')
    featurize = Conf['FEATS']
    num_feats = 0
    for i, nm in enumerate(proc_names):
        if os.path.exists(nm):
            t1 = imread(nm)
            tk1 = proc_keypoints[i]
            p1 = hr_predict_single(t1, tk1, hr_model_configs, overlap=0.5)
            if not isinstance(p1, list): p1 = [p1]
            num_feats = len(featurize.process(
                0, 0, t1[0:WINDOW, 0:WINDOW, :], t1, tk1, p1))
            break
    if not num_feats: return

    M, N, _ = t1.shape
    xsr, ysr = sampling(M, N, WINDOW)
    batch_samples = min(100, len(proc_names)) * len(xsr)
    train_x = np.zeros((batch_samples, num_feats), dtype=np.float)
    train_y = np.zeros((batch_samples,), dtype=np.uint8)
    name = os.path.join(tmdir,
                        'hair.{}.txt.{}'.format(os.getpid(), Set.lower()))
    print '[{}] # patches/img = {} Feature size = {}'.format(
        os.getpid(), len(xsr), num_feats)
    print '[{}] Writing to {}'.format(os.getpid(), name)
    fp = open(name, 'wb')
    j = 0
    np.random.seed(seed)

    for i, (fn, keyp) in enumerate(zip(proc_names, proc_keypoints)):
        if not os.path.exists(fn):
            print "Errr {} {}".format(i, fn)
            continue
        im = imread(fn)
        gt = img2gt(fn)
        m, n, _ = im.shape
        hr_preds = hr_predict_single(im, keyp, hr_model_configs, overlap=0.5)
        if not isinstance(hr_preds, list): hr_preds = [hr_preds]

        # sample patches
        if m != M or n != N:
            xsr, ysr = sampling(m, n, WINDOW)
            M, N = m, n
        if np.any(keyp[:-1, :] > np.asarray([m - WINDOW//2, n - WINDOW//2])):
            print '[{}] Skipping {}'.format(os.getpid(), fn)
            continue
        for x, y in zip(xsr, ysr):
            patch = im[y:y+WINDOW, x:x+WINDOW, :]
            gtpatch = gt[y:y+WINDOW, x:x+WINDOW]
            train_x[j, :] = featurize.process(x, y, patch, im, keyp, hr_preds)
            train_y[j] = featurize.processY(gtpatch)
            j += 1
        if j == train_x.shape[0]:
            dump_svmlight_file(train_x, train_y, fp)
            j = 0
        if i % 10 == 0:
            print "[{}] Done {}.".format(os.getpid(), i)

    if j: dump_svmlight_file(train_x[:j, :], train_y[:j], fp)
    fp.close()
    if out_q:
        out_q.put(name)
    else:
        return train_x, train_y


def count_samples(ys):
    n_face, n_hair, n_bkg = 0, 0, 0
    for e in ys:
        n_hair += (e==HAIR)
        n_face += (e==FACE)
        n_bkg += (e==BKG)
    return n_hair, n_face, n_bkg


def mat_to_name_keyp(mat_file):
    kp = loadmat(mat_file)
    keypoints = map(lambda k: k[0] - 1, kp['Keypoints'])
    names = map(lambda k: 'LFW/lfw_funneled/' + '/'.join(str(k[0][0]).split('/')[-2:]),
                kp['Names'])
    # hogs = map(lambda k: k[0], kp['HOGs'])
    return names, keypoints


def main():
    global Set, Conf, WINDOW
    args = argparse.ArgumentParser('Output: hair_<window>_<suf>.txt.<set>')
    args.add_argument('set', help='Train, Test or Validation')
    args.add_argument('suf', help='Feature config name.')
    args.add_argument('--hr', '-u', help='Hierarchical models. model_path1,...')
    args.add_argument('--out', '-o', help='Override output file name.')
    parse = args.parse_args()
    Conf = getattr(configs, parse.suf)()

    Set = parse.set
    WINDOW = Conf['WINDOW']
    outname = os.path.join(DATA_DIR, 'hair_{}.txt.{}'.format(
        parse.suf, Set.lower())) if not parse.out else parse.out

    print "Using {} set".format(Set)
    train_mat_path = 'LFW/FaceKeypointsHOG_11_{}.mat'.format(Set)
    names, keypoints = mat_to_name_keyp(train_mat_path)

    NUM_TRAIN_SAMPS = len(names)

    nprocs = mp.cpu_count()
    out_q = mp.Queue()
    chunksize = int(NUM_TRAIN_SAMPS // nprocs)
    procs = []
    rngs = np.random.randint(0, 100000, size=(nprocs,))

    print "Total training samples: {}. Starting {} procs with chunksize of {}.".format(
        NUM_TRAIN_SAMPS, nprocs, chunksize)

    for i in range(nprocs):
        hr_model_configs = hr_name_to_models(parse.hr)
        lim = chunksize * (i+1) if i < nprocs - 1 else NUM_TRAIN_SAMPS
        p = mp.Process(target=gen_train_data,
                       args=(names[chunksize*i:lim],
                             keypoints[chunksize*i:lim],
                             hr_model_configs,
                             out_q, rngs[i]))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    print "Joined processes"

    trainN = []
    for i in range(nprocs):
        n = out_q.get()
        trainN.append(n)

    # merge proc train files into 1
    cmd = ["cat"]
    cmd.extend(trainN)
    cmd.append('>')
    cmd.append(outname)
    call(' '.join(cmd), shell=True)

    print "Done"
    print outname

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print e
        raise
    # clean up
    cmd = ["rm", "-rf", tmdir]
    call(' '.join(cmd), shell=True)
